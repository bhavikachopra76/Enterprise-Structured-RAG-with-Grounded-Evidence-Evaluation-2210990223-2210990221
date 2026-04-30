"""
branch_selector.py
LLM-guided tree navigation for HierRAG. Fully vectorless.

Instead of embedding-based retrieval, this module asks Gemini to look at
the paper's section map (section IDs + summaries) and pick which sections
most likely contain the answer. This is the "navigation" step in the
HierRAG pipeline.

Returns three sets of chunks to the pipeline:
    primary_chunks  — leaf chunks from Gemini's top 1-2 section picks
    fallback_chunks — leaf chunks from a backup section (used if primary fails groundedness)
    flat_chunks     — every leaf chunk in the tree (tier 3 last-resort fallback)
"""

import json
from hierarchical_index import HierarchicalIndex, TreeNode
from gemini_client import call_gemini_json, sleep_between_calls


# navigation prompt — tells Gemini to pick sections, not generate an answer
_NAV_PROMPT = """You are navigating a research paper to find the answer to a question.
You are given a map of the paper's sections — each with an ID and a summary.

Your job: identify which sections most likely contain the answer.

Question: {query}

Paper map:
{tree_map}

Reply in this exact JSON format with no other text and no markdown fences:
{{
    "thinking": "one or two sentences explaining your reasoning",
    "primary": ["section_id_1"],
    "fallback": ["section_id_2"]
}}

Rules:
- primary: 1-2 section IDs most likely to contain the answer
- fallback: exactly 1 section ID to try if primary does not work
- primary and fallback must not overlap
- use only IDs that appear in the paper map above
- reply with JSON only, nothing else"""


class BranchSelector:
    """Routes questions to the right sections using Gemini as a navigator.

    Wraps a HierarchicalIndex and provides select_chunks() as the main
    entry point. The pipeline calls this once per question.
    """

    def __init__(self, index: HierarchicalIndex):
        self.index = index

    def select_chunks(self, query: str, root: TreeNode) -> dict:
        """Ask Gemini which sections to retrieve from, then collect their chunks.

        Returns:
            dict with keys:
                primary_chunks  : [TreeNode, ...]  all chunks from primary sections
                fallback_chunks : [TreeNode, ...]  all chunks from fallback section
                primary_ids     : [str, ...]       selected primary section IDs
                fallback_ids    : [str, ...]       selected fallback section IDs
                thinking        : str              Gemini's reasoning (useful for debugging)
                nav_response    : dict             raw Gemini response
        """
        tree_map   = self.index.get_tree_map(root)
        nav_result = self._call_gemini_navigation(query, tree_map)
        sleep_between_calls(0.5)

        primary_ids  = nav_result.get("primary",  [])
        fallback_ids = nav_result.get("fallback", [])

        # grab all leaf chunks from the selected sections
        primary_chunks  = self.index.get_chunks_from_sections(primary_ids, root)
        fallback_chunks = self.index.get_chunks_from_sections(fallback_ids, root)

        return {
            "primary_chunks"  : primary_chunks,
            "fallback_chunks" : fallback_chunks,
            "primary_ids"     : primary_ids,
            "fallback_ids"    : fallback_ids,
            "thinking"        : nav_result.get("thinking", ""),
            "nav_response"    : nav_result,
        }

    def flat_chunks(self, root: TreeNode) -> list:
        """Return every leaf chunk in the tree — tier 3 last-resort fallback.

        No navigation involved. The pipeline uses this when both tier 1 and
        tier 2 fail groundedness and we need to try with all available context.
        """
        return self.index.get_all_chunks(root)

    def _call_gemini_navigation(self, query: str, tree_map: dict) -> dict:
        """Call Gemini with the navigation prompt, validate the returned IDs.

        Handles several edge cases:
        - Filters out section IDs that don't exist in the tree map
        - Removes overlap between primary and fallback
        - Fills in defaults if Gemini returned empty lists
        - Falls back to first available sections if the call errors out entirely
        """
        prompt = _NAV_PROMPT.format(
            query    = query,
            tree_map = json.dumps(tree_map, indent=2),
        )

        try:
            result    = call_gemini_json(prompt, temperature=0.0)
            valid_ids = {s["id"] for s in tree_map["sections"]}

            # only keep IDs that actually exist in this paper
            result["primary"]  = [i for i in result.get("primary",  []) if i in valid_ids]
            result["fallback"] = [i for i in result.get("fallback", []) if i in valid_ids]

            # make sure primary and fallback don't overlap
            result["fallback"] = [i for i in result["fallback"] if i not in result["primary"]]

            # if Gemini returned nothing useful, default to the first sections
            secs = tree_map.get("sections", [])
            if not result["primary"] and secs:
                result["primary"] = [secs[0]["id"]]
            if not result["fallback"]:
                for sec in secs:
                    if sec["id"] not in result["primary"]:
                        result["fallback"] = [sec["id"]]
                        break

            return result

        except Exception as e:
            # total failure — just use the first two sections and move on
            print(f"  [selector] navigation failed: {e} — using first sections")
            secs = tree_map.get("sections", [])
            return {
                "thinking" : "Navigation failed, using first sections",
                "primary"  : [secs[0]["id"]] if len(secs) > 0 else [],
                "fallback" : [secs[1]["id"]] if len(secs) > 1 else [],
            }


# ── TEST ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=" * 55)
    print("branch_selector.py — running tests")
    print("=" * 55)

    with open("data/qasper/prepared_data.json") as f:
        docs = json.load(f)
    doc = docs[0]
    print(f"\nUsing: {doc['title'][:60]}\n")

    index    = HierarchicalIndex()
    root     = index.get_or_build_tree(doc)
    selector = BranchSelector(index)

    # test 1 — navigation
    print("Test 1: Gemini navigation")
    query  = doc["qa_pairs"][0]["question"]
    result = selector.select_chunks(query, root)

    assert "primary_chunks"  in result
    assert "fallback_chunks" in result
    assert "thinking"        in result

    print(f"  Query    : {query}")
    print(f"  Thinking : {result['thinking'][:100]}")
    print(f"  Primary  : {result['primary_ids']} → {len(result['primary_chunks'])} chunks")
    print(f"  Fallback : {result['fallback_ids']} → {len(result['fallback_chunks'])} chunks")
    print(f"  PASS\n")

    # test 2 — chunks are leaves with real content
    print("Test 2: chunk content quality")
    for i, chunk in enumerate(result["primary_chunks"][:3]):
        assert chunk.metadata.get("is_leaf")
        assert len(chunk.content.split()) >= 10, f"Fragment chunk: {chunk.content}"
        print(f"  chunk {i}: '{chunk.content[:80]}...'")
    print(f"  PASS\n")

    # test 3 — flat fallback
    print("Test 3: flat fallback")
    flat = selector.flat_chunks(root)
    assert len(flat) > 0
    print(f"  PASS — {len(flat)} total chunks in tree\n")

    # test 4 — all QA pairs
    print("Test 4: all QA pairs")
    for qa in doc["qa_pairs"]:
        r = selector.select_chunks(qa["question"], root)
        assert len(r["primary_chunks"]) > 0
        sleep_between_calls(0.5)
        print(f"  Q: {qa['question'][:60]}")
        print(f"     primary={r['primary_ids']} chunks={len(r['primary_chunks'])}")
    print(f"  PASS\n")

    print("=" * 55)
    print("All tests passed.")
    print("=" * 55)