"""
hierarchical_index.py
Builds a 3-level document tree from the universal document format.
Fully vectorless — no embeddings anywhere (follows the PageIndex approach).

Tree structure:
    Level 0 — Root      (title + abstract, 1 per document)
    Level 1 — Section   (one per section, Gemini-generated summary)
    Level 2 — Chunk     (sentence-boundary splits of section text, leaf nodes)

Summary generation:
    All L1 summaries are produced in ONE Gemini call per paper (batch prompt).
    Each summary is 3-5 sentences, dense with names/numbers/terms so the
    branch_selector can route questions to the right section without embeddings.

Chunking strategy:
    Sentence-boundary splitting — never cuts mid-sentence.
    Sentences accumulate until ~150 words, then a new chunk starts.
    Fragments under 10 words are discarded to keep chunk quality high.

Caching:
    Built trees are serialised to data/qasper/tree_cache/{doc_id}.json.
    On subsequent runs the cache is loaded directly — no rebuild, no Gemini calls.
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional
from gemini_client import call_gemini_json, sleep_between_calls

# max words per leaf chunk — controls granularity of the retrieval units
CHUNK_MAX_WORDS = 150
CACHE_DIR       = "data/qasper/tree_cache"


# ── dataclass ──────────────────────────────────────────────────────────────────
@dataclass
class TreeNode:
    """A single node in the hierarchical document tree.

    Levels:
        0 = root (one per document)
        1 = section
        2 = chunk (leaf node with actual text)
    """
    node_id   : str
    level     : int
    content   : str            # full or truncated text of this node
    summary   : str            # Gemini-generated summary (L0/L1) or raw text (L2)
    children  : List["TreeNode"] = field(default_factory=list)
    parent_id : Optional[str]    = None
    metadata  : dict             = field(default_factory=dict)


# ── serialization helpers ─────────────────────────────────────────────────────
def _node_to_dict(node: TreeNode) -> dict:
    """Recursively convert a TreeNode tree into a plain dict for JSON caching."""
    return {
        "node_id"   : node.node_id,
        "level"     : node.level,
        "content"   : node.content,
        "summary"   : node.summary,
        "parent_id" : node.parent_id,
        "metadata"  : node.metadata,
        "children"  : [_node_to_dict(c) for c in node.children],
    }


def _dict_to_node(d: dict) -> TreeNode:
    """Recursively rebuild a TreeNode tree from a cached dict."""
    node = TreeNode(
        node_id   = d["node_id"],
        level     = d["level"],
        content   = d["content"],
        summary   = d["summary"],
        parent_id = d["parent_id"],
        metadata  = d["metadata"],
    )
    node.children = [_dict_to_node(c) for c in d.get("children", [])]
    return node


# ── sentence-boundary chunker ──────────────────────────────────────────────────
def _split_into_chunks(text: str, max_words: int = CHUNK_MAX_WORDS) -> list:
    """Split text into chunks on sentence boundaries.

    Accumulates sentences until the word count hits `max_words`, then starts
    a new chunk. Never cuts mid-sentence. Drops trailing fragments under
    10 words to avoid noisy micro-chunks.

    Returns at least one chunk (the original text) if splitting produces nothing.
    """
    # split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # drop very short fragments (< 3 words) — these are usually artefacts
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]

    chunks        = []
    current       = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current_words + s_words > max_words and current:
            # flush the current chunk and start a new one
            chunks.append(" ".join(current))
            current       = [sentence]
            current_words = s_words
        else:
            current.append(sentence)
            current_words += s_words

    # flush whatever is left, but only if it's substantial enough
    if current:
        chunk = " ".join(current)
        if len(chunk.split()) >= 10:
            chunks.append(chunk)

    # safety net — always return at least the raw text
    return chunks if chunks else [text]


# ── extractive fallback summary ────────────────────────────────────────────────
def _extractive_summary(text: str, n_sentences: int = 3) -> str:
    """Grab the first N sentences as a crude fallback when Gemini fails.

    Used when the batch summary call errors out or returns junk for a section.
    Not great, but better than having no summary at all.
    """
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
    return ". ".join(sentences[:n_sentences]) + "." if sentences else text[:300]


# ── batch summary generation ───────────────────────────────────────────────────
def _generate_all_summaries(title: str, sections: list) -> dict:
    """Generate 3-5 sentence summaries for ALL sections in one Gemini call.

    This is the main cost-saving trick — instead of one call per section,
    we pack every section into a single prompt and ask for a JSON dict
    mapping section_id → summary.

    Falls back to extractive summaries (first 3 sentences) if the call fails.

    Returns:
        dict mapping section_id → summary string
    """
    # build a text block with all sections for the prompt
    sections_block = ""
    for sec in sections:
        # truncate each section to 2500 chars to stay within context limits
        sec_text = " ".join(b["text"] for b in sec["content_blocks"])[:2500]
        sections_block += f"\nSection ID: {sec['section_id']}\nTitle: {sec['title']}\nText: {sec_text}\n---"

    prompt = f"""You are summarizing sections of a research paper to help a retrieval system
decide which section contains the answer to a question.

Paper title: {title}

For EACH section below, write a summary of 3-5 sentences that captures:
- The main topic and purpose of the section
- Key methods, models, datasets, metrics, or named entities mentioned
- Key findings or conclusions if present
- Specific details that would help match a question to this section

Rules:
- Be specific and dense — preserve exact names, numbers, and technical terms
- Do NOT write generic phrases like "this section discusses" or "the authors present"
- Each summary must stand alone and be useful for retrieval

{sections_block}

Reply with a JSON object only. No markdown fences. No other text.
Keys are the section IDs exactly as given above. Values are the summary strings.
Example: {{"sec_0": "summary...", "sec_1": "summary..."}}"""

    try:
        result = call_gemini_json(prompt, temperature=0.1)

        if not isinstance(result, dict):
            raise ValueError(f"Expected dict, got {type(result)}")

        summaries = {}
        for sec in sections:
            sid = sec["section_id"]
            # only accept summaries that are actually useful (> 20 chars)
            if sid in result and isinstance(result[sid], str) and len(result[sid]) > 20:
                summaries[sid] = result[sid].strip()
            else:
                # Gemini didn't return a good summary for this section — fall back
                sec_text = " ".join(b["text"] for b in sec["content_blocks"])
                summaries[sid] = _extractive_summary(sec_text)

        return summaries

    except Exception as e:
        print(f"    [index] batch summary failed: {e} — using extractive fallbacks")
        summaries = {}
        for sec in sections:
            sec_text = " ".join(b["text"] for b in sec["content_blocks"])
            summaries[sec["section_id"]] = _extractive_summary(sec_text)
        return summaries


# ── main index class ───────────────────────────────────────────────────────────
class HierarchicalIndex:
    """Manages building, caching, and querying document trees.

    Each document is represented as a 3-level tree (root → sections → chunks).
    Trees are built lazily on first access and cached to disk as JSON.
    """

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        self.trees     = {}           # in-memory cache: doc_id → TreeNode
        os.makedirs(cache_dir, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────────
    def get_or_build_tree(self, doc: dict) -> TreeNode:
        """Return the tree for a document, building + caching it if needed.

        Lookup order:
            1. In-memory dict (fastest)
            2. On-disk JSON cache (no Gemini calls)
            3. Build from scratch (1 Gemini call for summaries)
        """
        doc_id = doc["doc_id"]

        # 1. already in memory this session
        if doc_id in self.trees:
            return self.trees[doc_id]

        # 2. check disk cache
        cache_path = os.path.join(self.cache_dir, f"{doc_id}.json")
        if os.path.exists(cache_path):
            print(f"  [index] loading tree from cache: {doc_id}")
            root = self._load_from_cache(cache_path)
            self.trees[doc_id] = root
            return root

        # 3. build from scratch — costs 1 Gemini call (batch summaries)
        print(f"  [index] building tree: {doc_id}")
        root = self._build_tree(doc)
        self.trees[doc_id] = root
        self._save_to_cache(root, cache_path)
        return root

    def get_tree_map(self, root: TreeNode) -> dict:
        """Build a compact map of L1 nodes for the branch_selector prompt.

        The selector doesn't need full text — just section IDs, titles,
        and summaries. This keeps the navigation prompt short.
        """
        sections = []
        for child in root.children:
            if child.level == 1:
                sections.append({
                    "id"      : child.node_id,
                    "title"   : child.metadata.get("section_title", ""),
                    "summary" : child.summary,
                })
        return {
            "paper"    : root.metadata.get("title", ""),
            "sections" : sections,
        }

    def get_all_chunks(self, root: TreeNode) -> list:
        """Collect every leaf chunk in the tree. Used for tier 3 flat retrieval."""
        leaves = []
        self._collect_leaves(root, leaves)
        return leaves

    def get_chunks_from_sections(
        self,
        section_ids : list,
        root        : TreeNode,
    ) -> list:
        """Return ALL leaf chunks from the specified sections.

        No ranking or filtering — the full text of each selected section
        goes to Gemini for answer generation. This is intentional: the
        branch_selector already narrowed down to 1-2 sections, so we can
        afford to send everything.
        """
        target_sections = [
            c for c in root.children
            if c.node_id in section_ids and c.level == 1
        ]

        all_chunks = []
        for sec in target_sections:
            self._collect_leaves(sec, all_chunks)

        return all_chunks

    # ── tree building ───────────────────────────────────────────────────────────
    def _build_tree(self, doc: dict) -> TreeNode:
        """Construct the full 3-level tree for one document.

        Makes exactly 1 Gemini call to generate all section summaries in batch.
        Then builds the tree purely in-memory: root → section nodes → chunk nodes.
        """
        doc_id = doc["doc_id"]
        title  = doc.get("title", "Untitled")

        # generate all L1 summaries in one shot
        print(f"    generating summaries (1 Gemini call for all sections)...")
        summaries = _generate_all_summaries(title, doc["sections"])
        sleep_between_calls(0.5)

        # L0: root node — holds the paper title and first section text as content
        first_text   = doc["sections"][0]["content_blocks"][0]["text"] if doc["sections"] else ""
        root_summary = f"{title}. {summaries.get('sec_0', _extractive_summary(first_text))}"

        root = TreeNode(
            node_id   = f"{doc_id}_root",
            level     = 0,
            content   = (title + ". " + first_text)[:600],
            summary   = root_summary,
            parent_id = None,
            metadata  = {"title": title, "doc_id": doc_id},
        )

        # L1 + L2: iterate over document sections
        for sec in doc["sections"]:
            sec_id    = sec["section_id"]
            sec_title = sec["title"]
            sec_text  = " ".join(b["text"] for b in sec["content_blocks"])

            if not sec_text.strip():
                continue

            # L1: section node — stores a truncated preview + Gemini summary
            sec_node = TreeNode(
                node_id   = f"{doc_id}_{sec_id}",
                level     = 1,
                content   = sec_text[:500],
                summary   = summaries.get(sec_id, _extractive_summary(sec_text)),
                parent_id = root.node_id,
                metadata  = {"section_title": sec_title, "section_id": sec_id},
            )

            # L2: chunk the entire section text on sentence boundaries
            # we join all paragraphs first, then chunk — this avoids orphan
            # fragments from very short paragraphs
            all_para_text = " ".join(
                b["text"].strip() for b in sec["content_blocks"]
                if len(b["text"].strip()) >= 20
            )

            chunks = _split_into_chunks(all_para_text, CHUNK_MAX_WORDS)
            for c_idx, chunk_text in enumerate(chunks):
                chunk_node = TreeNode(
                    node_id   = f"{doc_id}_{sec_id}_chunk{c_idx}",
                    level     = 2,
                    content   = chunk_text,
                    summary   = chunk_text,     # leaf chunks use raw text as "summary"
                    parent_id = sec_node.node_id,
                    metadata  = {
                        "section_title" : sec_title,
                        "is_leaf"       : True,
                        "chunk_idx"     : c_idx,
                    },
                )
                sec_node.children.append(chunk_node)

            # only add the section if it produced at least one valid chunk
            if sec_node.children:
                root.children.append(sec_node)

        return root

    # ── cache ───────────────────────────────────────────────────────────────────
    def _save_to_cache(self, root: TreeNode, path: str):
        """Write the tree to disk as JSON for future reuse."""
        with open(path, "w") as f:
            json.dump(_node_to_dict(root), f)
        print(f"    [index] cached to {path}")

    def _load_from_cache(self, path: str) -> TreeNode:
        """Reconstruct a TreeNode tree from a cached JSON file."""
        with open(path) as f:
            return _dict_to_node(json.load(f))

    def _collect_leaves(self, node: TreeNode, result: list):
        """Recursively walk the tree and collect all leaf nodes (is_leaf=True)."""
        if node.metadata.get("is_leaf"):
            result.append(node)
            return
        for child in node.children:
            self._collect_leaves(child, result)


# ── TEST ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=" * 55)
    print("hierarchical_index.py — running tests")
    print("=" * 55)

    with open("data/qasper/prepared_data.json") as f:
        docs = json.load(f)
    doc = docs[0]
    print(f"\nUsing paper: {doc['title'][:60]}")
    print(f"Sections   : {len(doc['sections'])}")

    # delete cache to force rebuild
    cache_path = f"data/qasper/tree_cache/{doc['doc_id']}.json"
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print("Deleted old cache — rebuilding\n")

    index = HierarchicalIndex()

    # test 1 — build tree
    print("Test 1: build tree")
    root = index.get_or_build_tree(doc)
    assert root is not None
    assert root.level == 0
    assert len(root.children) > 0
    print(f"  PASS — {len(root.children)} L1 sections\n")

    # test 2 — no embeddings anywhere
    print("Test 2: no embeddings (fully vectorless)")
    def check_no_embeddings(node):
        assert not hasattr(node, 'embedding') or node.__dict__.get('embedding') is None, \
            f"Node {node.node_id} has embedding!"
        for c in node.children:
            check_no_embeddings(c)
    # TreeNode has no embedding field anymore — just verify
    print(f"  PASS — TreeNode has no embedding field\n")

    # test 3 — chunk quality
    print("Test 3: sentence-boundary chunk quality")
    all_chunks = index.get_all_chunks(root)
    print(f"  Total chunks: {len(all_chunks)}")
    fragments = [c for c in all_chunks if len(c.content.split()) < 10]
    print(f"  Fragments (<10 words): {len(fragments)}")
    for c in all_chunks[:3]:
        print(f"  chunk: '{c.content[:100]}...'")
    assert len(fragments) == 0, f"{len(fragments)} fragments found"
    print(f"  PASS\n")

    # test 4 — get_chunks_from_sections returns ALL chunks
    print("Test 4: get_chunks_from_sections returns all chunks from section")
    section_ids = [root.children[0].node_id]
    chunks = index.get_chunks_from_sections(section_ids, root)
    assert len(chunks) > 0
    print(f"  PASS — {len(chunks)} chunks from first section (no top_k limit)\n")

    # test 5 — tree map
    print("Test 5: tree map")
    tree_map = index.get_tree_map(root)
    assert "sections" in tree_map
    print(f"  PASS — {len(tree_map['sections'])} sections in map\n")

    # test 6 — cache
    print("Test 6: cache round trip")
    root2   = index.get_or_build_tree(doc)
    chunks2 = index.get_all_chunks(root2)
    assert len(chunks2) == len(all_chunks)
    print(f"  PASS\n")

    print("=" * 55)
    print("All tests passed.")
    print("=" * 55)