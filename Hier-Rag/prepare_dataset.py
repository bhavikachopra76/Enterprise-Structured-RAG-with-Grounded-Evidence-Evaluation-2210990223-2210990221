"""
prepare_dataset.py
Loads QASPER from HuggingFace, converts it to HierRAG's universal document
format, and saves to data/qasper/prepared_data.json.

Universal document format (what the tree builder reads):
{
    "doc_id":   str,
    "title":    str,
    "sections": [
        {
            "section_id":     str,          # e.g. "sec_0"
            "title":          str,          # section heading
            "level":          int,          # always 1 for QASPER (flat sections)
            "content_blocks": [
                {"type": "paragraph", "text": str},
                ...
            ]
        }
    ],
    "qa_pairs": [
        {"question": str, "answer": str, "evidence": [str]}
    ]
}

This format is source-agnostic — PDF and DOCX parsers can output the same
structure so the tree builder never needs to know where the data came from.

Filters applied during conversion:
    - Skips papers with no sections or no usable QA pairs
    - Skips yes/no, "Unanswerable", and empty answers
    - Skips answers that are purely BIBREF citation tokens
    - Skips answers where > 50% of tokens are BIBREFs
    - Drops paragraphs under 30 characters (noise)
"""

import json
import os
import re
from datasets import load_dataset
from tqdm import tqdm


# ── QASPER → universal adapter ─────────────────────────────────────────────────
def qasper_paper_to_universal(paper: dict, paper_idx: int) -> dict | None:
    """Convert one raw QASPER paper dict into the universal document format.

    Returns None if the paper has no usable sections or QA pairs, so
    the caller can skip it without crashing.
    """
    paper_id = paper.get("id") or f"paper_{paper_idx}"
    title    = paper.get("title", "Untitled")
    abstract = paper.get("abstract", "")

    # ── build sections ──────────────────────────────────────────────────────────
    sections = []

    # abstract always becomes section 0
    if abstract.strip():
        sections.append({
            "section_id": "sec_0",
            "title": "Abstract",
            "level": 1,
            "content_blocks": [{"type": "paragraph", "text": abstract.strip()}]
        })

    # remaining sections from the full_text field
    full_text = paper.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    paragraphs_by_section = full_text.get("paragraphs") or []

    for i, section_name in enumerate(section_names):
        if i >= len(paragraphs_by_section):
            break

        raw_paras = paragraphs_by_section[i]

        # only keep paragraphs with real content (>= 30 chars)
        blocks = []
        for para in raw_paras:
            if para and len(para.strip()) >= 30:
                blocks.append({"type": "paragraph", "text": para.strip()})

        if not blocks:
            continue

        sections.append({
            "section_id": f"sec_{len(sections)}",
            "title": section_name.strip() if section_name else f"Section {i+1}",
            "level": 1,
            "content_blocks": blocks
        })

    if not sections:
        return None

    # ── build qa pairs ──────────────────────────────────────────────────────────
    qa_pairs = []
    qas = paper.get("qas") or {}
    questions = qas.get("question") or []
    answers_list = qas.get("answers") or []

    for q_idx, question in enumerate(questions):
        if q_idx >= len(answers_list):
            break

        answer_data = answers_list[q_idx]
        answer_text = ""
        evidence    = []

        # dig through the nested answer structure — QASPER has multiple
        # annotator answers per question, we take the first usable one
        if answer_data and "answer" in answer_data:
            for ans in answer_data["answer"]:
                if ans.get("free_form_answer", "").strip():
                    answer_text = ans["free_form_answer"].strip()
                    evidence    = ans.get("evidence") or []
                    break
                elif ans.get("extractive_spans"):
                    answer_text = " ".join(ans["extractive_spans"]).strip()
                    evidence    = ans.get("evidence") or []
                    break

        # skip yes/no, unanswerable, and empty answers — they're not useful
        # for evaluating our retrieval pipeline
        if not answer_text or answer_text in {"Yes", "No", "Unanswerable"}:
            continue

        # skip answers that are purely BIBREF tokens (citation-only answers)
        if re.fullmatch(r'(BIBREF\d+\s*)+', answer_text.strip()):
            continue

        # skip answers where > 50% of tokens are BIBREFs (mixed garbage)
        tokens = answer_text.strip().split()
        bibref_tokens = [t for t in tokens if re.match(r'BIBREF\d+', t)]
        if len(tokens) > 0 and len(bibref_tokens) / len(tokens) > 0.5:
            continue

        qa_pairs.append({
            "question": question.strip(),
            "answer":   answer_text,
            "evidence": evidence
        })

    if not qa_pairs:
        return None

    return {
        "doc_id":   paper_id,
        "title":    title,
        "sections": sections,
        "qa_pairs": qa_pairs
    }


# ── main preparation ────────────────────────────────────────────────────────────
def prepare_qasper(max_papers: int = 200, max_qa_per_paper: int = 3) -> list[dict]:
    """Load QASPER validation set, convert to universal format, return docs.

    Uses HuggingFace's auto-converted Parquet files to bypass the deprecated
    qasper.py loading script that breaks on newer datasets versions.

    Args:
        max_papers:       cap on number of papers to include
        max_qa_per_paper: cap on QA pairs per paper (keeps eval cost low)

    Returns:
        List of universal document dicts ready for the tree builder.
    """
    print("Loading QASPER dataset from HuggingFace...")
    # load from Parquet to avoid the deprecated qasper.py script
    dataset = load_dataset(
        "parquet", 
        data_files={"validation": "hf://datasets/allenai/qasper@refs/convert/parquet/qasper/validation/*.parquet"}
    )
    val_data = dataset["validation"]
    print(f"  Validation set size: {len(val_data)} papers")

    prepared = []
    skipped  = 0

    for idx, paper in enumerate(tqdm(val_data, desc="Converting papers")):
        if len(prepared) >= max_papers:
            break

        doc = qasper_paper_to_universal(paper, idx)
        if doc is None:
            skipped += 1
            continue

        # cap QA pairs per paper to control experiment cost
        doc["qa_pairs"] = doc["qa_pairs"][:max_qa_per_paper]
        prepared.append(doc)

    print(f"\n  Converted : {len(prepared)} papers")
    print(f"  Skipped   : {skipped} papers (no sections or QA pairs)")
    print(f"  Total QAs : {sum(len(d['qa_pairs']) for d in prepared)}")
    return prepared


# ── TEST ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("prepare_dataset.py — running tests")
    print("=" * 55)

    # test 1 — small load (20 papers) to verify structure
    print("\nTest 1: load 20 papers and check universal format")
    docs = prepare_qasper(max_papers=20, max_qa_per_paper=3)

    assert len(docs) > 0, "No documents returned"

    doc = docs[2]
    print(f"\n  First doc:")
    print(f"    doc_id   : {doc['doc_id']}")
    print(f"    title    : {doc['title'][:60]}")
    print(f"    sections : {len(doc['sections'])}")
    print(f"    qa_pairs : {len(doc['qa_pairs'])}")

    # check universal format keys
    assert "doc_id"   in doc, "Missing doc_id"
    assert "title"    in doc, "Missing title"
    assert "sections" in doc, "Missing sections"
    assert "qa_pairs" in doc, "Missing qa_pairs"
    print("  PASS — top-level keys OK")

    # check section format
    sec = doc["sections"][0]
    assert "section_id"     in sec, "Missing section_id"
    assert "title"          in sec, "Missing section title"
    assert "level"          in sec, "Missing level"
    assert "content_blocks" in sec, "Missing content_blocks"
    assert len(sec["content_blocks"]) > 0, "Section has no content blocks"
    assert "type" in sec["content_blocks"][0], "content_block missing type"
    assert "text" in sec["content_blocks"][0], "content_block missing text"
    print("  PASS — section format OK")

    # check QA format
    qa = doc["qa_pairs"][0]
    assert "question" in qa, "Missing question"
    assert "answer"   in qa, "Missing answer"
    assert "evidence" in qa, "Missing evidence"
    assert qa["answer"] not in {"Yes", "No", "Unanswerable"}, "Yes/No answer slipped through"
    print("  PASS — QA pair format OK")

    # test 2 — print one full section so you can visually verify
    print("\nTest 2: visual check — first section of first doc")
    sec0 = doc["sections"][0]
    print(f"  section_id : {sec0['section_id']}")
    print(f"  title      : {sec0['title']}")
    print(f"  blocks     : {len(sec0['content_blocks'])}")
    print(f"  first block (first 120 chars):")
    print(f"    {sec0['content_blocks'][0]['text'][:120]}...")

    # test 3 — save to disk
    print("\nTest 3: save to disk")
    os.makedirs("data/qasper", exist_ok=True)
    out_path = "data/qasper/prepared_data.json"
    with open(out_path, "w") as f:
        json.dump(docs, f, indent=2)
    assert os.path.exists(out_path), "File was not created"
    file_size_kb = os.path.getsize(out_path) / 1024
    print(f"  PASS — saved to {out_path} ({file_size_kb:.1f} KB)")

    # test 4 — reload and verify round-trip
    print("\nTest 4: reload from disk and verify round-trip")
    with open(out_path) as f:
        reloaded = json.load(f)
    assert len(reloaded) == len(docs), "Length mismatch after reload"
    assert reloaded[0]["doc_id"] == docs[0]["doc_id"], "doc_id mismatch after reload"
    print("  PASS — round-trip OK")

    print("\n" + "=" * 55)
    print("All tests passed. Run with max_papers=200 for full dataset.")
    print("=" * 55)

    # ── full run ────────────────────────────────────────────────────────────────
    print("\nRunning full preparation (100 papers)...")
    full_docs = prepare_qasper(max_papers=100, max_qa_per_paper=3)
    with open("data/qasper/prepared_data.json", "w") as f:
        json.dump(full_docs, f, indent=2)
    print(f"Saved {len(full_docs)} papers to data/qasper/prepared_data.json")