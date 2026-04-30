"""
sensitivity_tester.py
Context-removal sensitivity testing for HierRAG.

Measures how much each retrieved chunk actually matters to the generated answer.
For each chunk, we re-generate the answer without it and compare to the baseline.

Sensitivity formula:
    S(chunk_i) = 1 - cosine_similarity(baseline_answer, answer_without_chunk_i)

    High S → removing that chunk changed the answer a lot → chunk is load-bearing
    Low S  → answer barely changed → chunk was noise or redundant

Why cosine similarity instead of F1-delta against ground truth:
    F1-delta measures answer quality, not chunk influence. Two answers can be
    equally correct but phrased differently — F1 stays the same even though
    the chunk changed the output completely. Cosine similarity between sentence
    embeddings captures whether the answer CONTENT shifted, regardless of labels.
    This makes sensitivity an unsupervised signal that doesn't need ground truth.

    temperature=0 on all re-generation calls removes LLM randomness from the
    measurement entirely, so any change in output is attributable to the
    missing chunk, not sampling variance.

Note: ground_truth is still accepted for F1/EM reporting but is NOT used in the
      sensitivity calculation itself.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from gemini_client import call_gemini, sleep_between_calls


# ── embedding model ─────────────────────────────────────────────────────────────
# default: all-MiniLM-L6-v2 — fast, well-cited in RAG literature, ~80MB
# swap via set_embed_model() before the first test() call if you need another
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    """Lazy-load the sentence embedding model. Only loaded once per process."""
    global _embed_model
    if _embed_model is None:
        print(f"  [sensitivity] loading embedding model: {_EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def set_embed_model(model_name: str):
    """Switch the embedding model. Must be called before the first test() call.

    Resets the cached model so the next get_embed_model() loads the new one.
    """
    global _EMBED_MODEL_NAME, _embed_model
    _EMBED_MODEL_NAME = model_name
    _embed_model      = None   # force reload on next access


# ── generation prompt ────────────────────────────────────────────────────────────
# same prompt as hierrag_pipeline uses — keeps answer format consistent
_GEN_PROMPT = """Answer the following question using only the provided context passages.
Be concise and specific. If the context does not contain the answer, say "Not found in context".

Question: {query}

Context:
{context}

Answer:"""


# ── text normalization for F1 / EM ─────────────────────────────────────────────
def _normalize(text: str) -> str:
    """Normalize text for token-level F1 and exact match computation.

    Strips LaTeX commands, BIBREF tokens, and punctuation. Lowercases everything.
    Only used for F1/EM metrics — NOT used in the sensitivity calculation.
    """
    # flatten LaTeX math: $x_{\\rm foo}$ → just the letters/numbers inside
    text = re.sub(r'\$[^\$]*\$', lambda m: re.sub(r'[\\{}]', ' ', m.group()), text)
    # remove remaining LaTeX commands like \\textbf
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    # strip BIBREF tokens — they're citation markers, not content
    text = re.sub(r'BIBREF\d+', ' ', text)
    # lowercase + strip non-alphanumeric
    text = re.sub(r'[^a-z0-9 ]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth (after normalization)."""
    pred_tokens  = set(_normalize(prediction).split())
    truth_tokens = set(_normalize(ground_truth).split())
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(truth_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 if normalized prediction exactly matches normalized ground truth, else 0.0."""
    return float(_normalize(prediction) == _normalize(ground_truth))


# ── cosine similarity via sentence embeddings ────────────────────────────────────
def _cosine_similarity(text_a: str, text_b: str) -> float:
    """Embed both strings and return cosine similarity in [0, 1].

    Uses normalized embeddings so dot product == cosine similarity (no extra math).
    """
    model = get_embed_model()
    embs  = model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
    sim   = float(np.dot(embs[0], embs[1]))
    return round(max(0.0, min(1.0, sim)), 4)   # clamp to [0, 1] for safety


# ── result dataclass ─────────────────────────────────────────────────────────────
@dataclass
class SensitivityResult:
    """Holds the outcome of a sensitivity test.

    avg_sensitivity = -1.0 means the test was skipped (e.g. flat retrieval).
    most_influential_idx points to the chunk whose removal changed the answer most.
    """
    avg_sensitivity      : float
    most_influential_idx : int
    chunk_sensitivities  : list  = field(default_factory=list)
    baseline_f1          : float = 0.0
    baseline_em          : float = 0.0

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON output."""
        return {
            "avg_sensitivity"      : round(self.avg_sensitivity, 4),
            "most_influential_idx" : self.most_influential_idx,
            "baseline_f1"          : round(self.baseline_f1, 4),
            "baseline_em"          : round(self.baseline_em, 4),
            "chunk_sensitivities"  : self.chunk_sensitivities,
        }


# ── main tester ──────────────────────────────────────────────────────────────────
class SensitivityTester:
    """Runs leave-one-out sensitivity analysis on retrieved chunks.

    For each chunk, removes it from the context, re-generates the answer
    with Gemini (temperature=0), and measures how much the answer changed
    using cosine similarity of sentence embeddings.
    """

    def skip(
        self,
        baseline_ans : str,
        ground_truth : str = "",
        reason       : str = "skipped",
    ) -> "SensitivityResult":
        """Return a placeholder result without making any Gemini calls.

        Used when sensitivity testing is too expensive (e.g. flat retrieval
        with dozens of chunks would mean dozens of extra Gemini calls).
        avg_sensitivity = -1 signals "not tested".
        """
        baseline_f1 = compute_f1(baseline_ans, ground_truth) if ground_truth else 0.0
        baseline_em = compute_exact_match(baseline_ans, ground_truth) if ground_truth else 0.0
        return SensitivityResult(
            avg_sensitivity      = -1.0,
            most_influential_idx = -1,
            baseline_f1          = baseline_f1,
            baseline_em          = baseline_em,
            chunk_sensitivities  = [{"note": reason}],
        )

    def test(
        self,
        question     : str,
        chunks       : list,
        baseline_ans : str,
        ground_truth : str = "",
    ) -> SensitivityResult:
        """Run leave-one-out sensitivity test.

        For each chunk i:
            1. Re-generate the answer WITHOUT chunk i (temperature=0)
            2. sensitivity(i) = 1 - cosine_sim(baseline_ans, answer_without_i)

        This costs len(chunks) Gemini calls — one per chunk removal.

        ground_truth is used only for F1/EM reporting in the result dict,
        not for the sensitivity calculation itself.
        """
        baseline_f1 = compute_f1(baseline_ans, ground_truth)          if ground_truth else 0.0
        baseline_em = compute_exact_match(baseline_ans, ground_truth)  if ground_truth else 0.0

        # edge case — only one chunk, so removing it leaves nothing
        if len(chunks) <= 1:
            return SensitivityResult(
                avg_sensitivity      = 1.0,
                most_influential_idx = 0,
                baseline_f1          = baseline_f1,
                baseline_em          = baseline_em,
                chunk_sensitivities  = [{
                    "chunk_idx"      : 0,
                    "sensitivity"    : 1.0,
                    "similarity"     : 0.0,
                    "answer_without" : "",
                    "chunk_preview"  : chunks[0][:100] + "...",
                    "note"           : "single chunk — sensitivity trivially 1.0",
                }],
            )

        chunk_sensitivities = []

        for i, chunk in enumerate(chunks):
            # build context without chunk i
            remaining = [c for j, c in enumerate(chunks) if j != i]
            context   = "\n\n".join(remaining)
            prompt    = _GEN_PROMPT.format(query=question, context=context)

            try:
                # temperature=0 — deterministic, so any output change is from the missing chunk
                ans_without = call_gemini(prompt, temperature=0.0)
            except Exception as e:
                print(f"  [sensitivity] call failed for chunk {i}: {e}")
                ans_without = ""

            sleep_between_calls(5.0)

            # compare the baseline answer to the answer-without-chunk-i
            similarity  = _cosine_similarity(baseline_ans, ans_without)
            sensitivity = round(1.0 - similarity, 4)

            chunk_sensitivities.append({
                "chunk_idx"      : i,
                "sensitivity"    : sensitivity,
                "similarity"     : similarity,
                "answer_without" : ans_without,
                "chunk_preview"  : chunk[:100] + "...",
            })

        # aggregate: mean sensitivity and which chunk mattered most
        sensitivities        = [c["sensitivity"] for c in chunk_sensitivities]
        avg_sensitivity      = round(float(np.mean(sensitivities)), 4)
        most_influential_idx = int(np.argmax(sensitivities))

        return SensitivityResult(
            avg_sensitivity      = avg_sensitivity,
            most_influential_idx = most_influential_idx,
            baseline_f1          = baseline_f1,
            baseline_em          = baseline_em,
            chunk_sensitivities  = chunk_sensitivities,
        )


# ── TEST ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("sensitivity_tester.py — running tests")
    print("=" * 55)

    tester = SensitivityTester()

    # test 1 — normalization
    print("\nTest 1: normalization")
    assert _normalize("pivoting pivoting$_{\\rm m}$") == "pivoting pivoting m m"
    assert _normalize("BIBREF19 BIBREF20") == ""
    assert _normalize("Europarl MultiUN")  == "europarl multiun"
    assert _normalize("OntoNotes 5.0")     == "ontonotes 5 0"
    print("  PASS\n")

    # test 2 — cosine similarity, identical strings
    print("Test 2: cosine similarity — identical strings")
    sim = _cosine_similarity(
        "The model was trained on OntoNotes 5.0",
        "The model was trained on OntoNotes 5.0",
    )
    print(f"  similarity (identical) : {sim}")
    assert sim > 0.99, f"Identical strings should have sim ~1.0, got {sim}"
    print("  PASS\n")

    # test 3 — cosine similarity, very different strings
    print("Test 3: cosine similarity — very different strings")
    sim = _cosine_similarity(
        "The model was trained on OntoNotes 5.0 using AdaGrad optimizer",
        "Not found in context",
    )
    print(f"  similarity (different) : {sim}")
    assert sim < 0.6, f"Very different strings should have low sim, got {sim}"
    print("  PASS\n")

    # test 4 — sensitivity direction
    print("Test 4: sensitivity direction")
    sim_high = _cosine_similarity(
        "The authors used AdaGrad with lr=0.001",
        "AdaGrad optimizer was used with learning rate 0.001",
    )
    sim_low = _cosine_similarity(
        "The authors used AdaGrad with lr=0.001",
        "Not found in context",
    )
    sens_low  = round(1.0 - sim_high, 4)
    sens_high = round(1.0 - sim_low,  4)
    print(f"  similar answers   → sensitivity = {sens_low}  (should be low)")
    print(f"  different answers → sensitivity = {sens_high} (should be high)")
    assert sens_low < sens_high
    print("  PASS\n")

    # test 5 — single chunk edge case
    print("Test 5: single chunk edge case")
    result = tester.test(
        question     = "What dataset was used?",
        chunks       = ["The model was trained on OntoNotes 5.0"],
        baseline_ans = "OntoNotes 5.0",
        ground_truth = "OntoNotes 5.0",
    )
    assert result.avg_sensitivity      == 1.0
    assert result.most_influential_idx == 0
    print(f"  avg_sensitivity : {result.avg_sensitivity}")
    print("  PASS\n")

    # test 6 — real Gemini sensitivity test
    print("Test 6: real sensitivity test (3 Gemini calls, temperature=0)")
    chunks = [
        "The authors evaluated their model on the OntoNotes 5.0 benchmark dataset.",
        "The CoNLL-2012 shared task data was used for training with 2802 documents.",
        "The model used AdaGrad optimizer with learning rate 0.001 during training.",
    ]
    result = tester.test(
        question     = "What dataset was used for evaluation?",
        chunks       = chunks,
        baseline_ans = "OntoNotes 5.0 benchmark dataset",
        ground_truth = "OntoNotes 5.0",
    )
    assert len(result.chunk_sensitivities) == 3
    print(f"  baseline F1      : {result.baseline_f1}")
    print(f"  avg sensitivity  : {result.avg_sensitivity}")
    print(f"  most influential : chunk {result.most_influential_idx}")
    for cs in result.chunk_sensitivities:
        print(
            f"  chunk {cs['chunk_idx']}: "
            f"sensitivity={cs['sensitivity']:.4f}  "
            f"similarity={cs['similarity']:.4f}  "
            f"'{cs['answer_without'][:60]}'"
        )
    print("  PASS\n")

    # test 7 — to_dict format
    print("Test 7: to_dict format")
    d = result.to_dict()
    for key in ["avg_sensitivity", "most_influential_idx", "baseline_f1",
                "baseline_em", "chunk_sensitivities"]:
        assert key in d, f"Missing key: {key}"
    for cs in d["chunk_sensitivities"]:
        for key in ["sensitivity", "similarity", "answer_without", "chunk_preview"]:
            assert key in cs, f"Missing chunk key: {key}"
    print(f"  dict keys: {list(d.keys())}")
    print("  PASS\n")

    print("=" * 55)
    print("All tests passed.")
    print("=" * 55)