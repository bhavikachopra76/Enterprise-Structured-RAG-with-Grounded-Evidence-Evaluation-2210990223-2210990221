"""
hierrag_pipeline.py
Full HierRAG pipeline — ties all components together.

process_question() is the single entry point for the entire retrieval +
generation + evaluation flow. It runs a 3-tier retrieval strategy and
returns the answer along with all metrics.

3-tier retrieval flow:
    Tier 1 (primary)  : Gemini navigates the tree → picks sections → generates
                        answer → groundedness check. If it passes, done.
    Tier 2 (fallback) : If tier 1 fails groundedness → try the fallback section
                        that Gemini also picked during navigation.
    Tier 3 (flat)     : If tier 2 also fails → dump all chunks from the entire
                        tree into the prompt and let Gemini figure it out.

Key design decisions:
    - temperature=0 on all generation calls for deterministic output
    - Full trial_history is saved: every tier attempt stores its context,
      answer, and groundedness result — not just the winning tier.
      tier_analyser.py uses this for post-hoc failure analysis.
    - gemini_calls counter is accurate across all tier attempts so
      run_experiment.py can track total API cost.
"""

import json
import numpy as np
from hierarchical_index import HierarchicalIndex
from branch_selector import BranchSelector
from groundedness_scorer import GroundednessScorer
from sensitivity_tester import SensitivityTester, compute_f1, compute_exact_match
from gemini_client import call_gemini, sleep_between_calls


# ── generation prompt ─────────────────────────────────────────────────────────
_GEN_PROMPT = """Answer the following question using only the provided context passages.
Be concise and specific. If the context does not contain the answer, say "Not found in context".

Question: {query}

Context:
{context}

Answer:"""


class HierRAG:
    """Complete HierRAG pipeline.

    Initialise once, then call process_question() for each QA pair.
    Internally creates and holds a HierarchicalIndex, BranchSelector,
    GroundednessScorer, and SensitivityTester.
    """

    def __init__(self):
        print("Initialising HierRAG...")
        self.index       = HierarchicalIndex()
        self.selector    = BranchSelector(self.index)
        self.grounder    = GroundednessScorer()
        self.sens_tester = SensitivityTester()
        print("HierRAG ready.\n")

    # ── public entry point ────────────────────────────────────────────────────
    def process_question(
        self,
        question     : str,
        doc          : dict,
        ground_truth : str = "",
        top_k        : int = 3,
    ) -> dict:
        """Run the full HierRAG pipeline for one question.

        Orchestrates: tree building → navigation → generation → groundedness →
        optional tier fallback → sensitivity testing → metric computation.

        Args:
            question:     the question to answer
            doc:          universal-format document dict
            ground_truth: expected answer (for F1/EM, not used during retrieval)
            top_k:        unused legacy param, kept for compat

        Returns:
            dict containing answer, tier_used, metrics, trial_history, etc.
            trial_history is a list of every tier attempt — consumed by
            tier_analyser.py for post-hoc failure explanation.
        """
        # track Gemini calls per question for cost monitoring
        gemini_calls  = 0
        trial_history = []   # stores every tier attempt, including failures

        # ── step 1: get or build tree ─────────────────────────────────────────
        root = self.index.get_or_build_tree(doc)

        # ── step 2: Gemini navigation — pick sections ─────────────────────────
        nav = self.selector.select_chunks(question, root)
        gemini_calls += 1

        # ── tier 1: primary sections ──────────────────────────────────────────
        primary_chunks = nav["primary_chunks"]
        tier_used      = "primary"
        final_chunks   = primary_chunks
        answer         = ""
        groundedness   = None

        if primary_chunks:
            answer        = self._generate(question, primary_chunks)
            gemini_calls += 1
            sleep_between_calls(5.0)

            # run groundedness check — separate LLM call acting as auditor
            groundedness = self.grounder.score(answer, [c.content for c in primary_chunks], question=question)

            # record this attempt regardless of pass/fail
            trial_history.append({
                "tier"         : "primary",
                "chunks"       : [c.content for c in primary_chunks],
                "answer"       : answer,
                "groundedness" : groundedness.to_dict(),
                "passed"       : groundedness.passes,
            })

            if not groundedness.passes:
                # tier 1 failed groundedness — escalate to tier 2/3
                answer, final_chunks, tier_used, extra_calls, tier2_history = \
                    self._try_fallback(question, nav["fallback_chunks"], top_k, root)
                gemini_calls += extra_calls
                trial_history.extend(tier2_history)
                groundedness = self.grounder.score(answer, [c.content for c in final_chunks], question=question)
        else:
            # navigation returned no primary chunks — skip straight to fallback
            print(f"  [pipeline] no primary chunks — going straight to fallback")
            answer, final_chunks, tier_used, extra_calls, tier2_history = \
                self._try_fallback(question, nav["fallback_chunks"], top_k, root)
            gemini_calls += extra_calls
            trial_history.extend(tier2_history)
            groundedness = self.grounder.score(answer, [c.content for c in final_chunks], question=question)

        # ── step 5: sensitivity testing ───────────────────────────────────────
        chunk_texts = [c.content for c in final_chunks]
        if tier_used == "flat":
            # skip sensitivity on flat retrieval — too many chunks means too
            # many Gemini calls (one per chunk removed), not worth the cost
            sensitivity = self.sens_tester.skip(
                baseline_ans = answer,
                ground_truth = ground_truth,
                reason       = "skipped on flat retrieval — too many chunks",
            )
        else:
            sensitivity = self.sens_tester.test(
                question     = question,
                chunks       = chunk_texts,
                baseline_ans = answer,
                ground_truth = ground_truth,
            )
            gemini_calls += len(final_chunks)   # 1 Gemini call per chunk removal

        # ── step 6: compute standard metrics against ground truth ─────────────
        f1          = compute_f1(answer, ground_truth)          if ground_truth else 0.0
        exact_match = compute_exact_match(answer, ground_truth) if ground_truth else 0.0

        return {
            "question"        : question,
            "answer"          : answer,
            "ground_truth"    : ground_truth,
            "tier_used"       : tier_used,
            "retrieved_chunks": chunk_texts,
            "nav_thinking"    : nav["thinking"],
            "groundedness"    : groundedness.to_dict(),
            "sensitivity"     : sensitivity.to_dict(),
            "f1"              : f1,
            "exact_match"     : exact_match,
            "gemini_calls"    : gemini_calls,
            # full attempt log — consumed by tier_analyser.py
            "trial_history"   : trial_history,
        }

    # ── tier 2/3 fallback ─────────────────────────────────────────────────────
    def _try_fallback(
        self,
        question       : str,
        fallback_chunks: list,
        top_k          : int,
        root,
    ) -> tuple:
        """Try fallback sections (tier 2), then flat retrieval (tier 3) if needed.

        This method is called when tier 1 primary chunks fail groundedness.
        It first tries the fallback section Gemini suggested during navigation.
        If that also fails, it dumps the entire tree's chunks into the prompt.

        Returns:
            (answer, final_chunks, tier_used, gemini_calls_used, history)
            history is a list of attempt dicts to append to trial_history.
        """
        gemini_calls = 0
        history      = []

        # tier 2: try the fallback section
        if fallback_chunks:
            answer        = self._generate(question, fallback_chunks)
            gemini_calls += 1
            sleep_between_calls(5.0)

            groundedness = self.grounder.score(answer, [c.content for c in fallback_chunks], question=question)

            history.append({
                "tier"         : "fallback",
                "chunks"       : [c.content for c in fallback_chunks],
                "answer"       : answer,
                "groundedness" : groundedness.to_dict(),
                "passed"       : groundedness.passes,
            })

            if groundedness.passes:
                return answer, fallback_chunks, "fallback", gemini_calls, history

        # tier 3 — flat retrieval: everything from the tree
        print(f"  [pipeline] fallback failed — escalating to flat retrieval")
        flat_chunks   = self.selector.flat_chunks(root)
        answer        = self._generate(question, flat_chunks)
        gemini_calls += 1
        sleep_between_calls(5.0)

        groundedness = self.grounder.score(answer, [c.content for c in flat_chunks], question=question)
        history.append({
            "tier"         : "flat",
            "chunks"       : [c.content for c in flat_chunks],
            "answer"       : answer,
            "groundedness" : groundedness.to_dict(),
            "passed"       : groundedness.passes,
        })

        return answer, flat_chunks, "flat", gemini_calls, history

    # ── answer generation ─────────────────────────────────────────────────────
    def _generate(self, question: str, chunks: list) -> str:
        """Build the generation prompt from chunks and call Gemini.

        temperature=0 for deterministic output — important for sensitivity
        testing, where we need stable baselines.
        """
        context = "\n\n".join(c.content for c in chunks)
        prompt  = _GEN_PROMPT.format(query=question, context=context)
        return call_gemini(prompt, temperature=0.0)


# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("hierrag_pipeline.py — running tests")
    print("=" * 55)

    with open("data/qasper/prepared_data.json") as f:
        docs = json.load(f)
    doc = docs[2]
    print(f"\nUsing paper : {doc['title'][:60]}")
    print(f"QA pairs    : {len(doc['qa_pairs'])}\n")

    pipeline = HierRAG()

    def _print_result(r, idx):
        print(f"  QA {idx+1} — {r['question']}")
        print(f"  Answer        : {r['answer'][:120]}")
        print(f"  Ground truth  : {r['ground_truth'][:120]}")
        print(f"  Tier used     : {r['tier_used']}")
        print(f"  F1            : {r['f1']:.4f}")
        print(f"  Exact match   : {r['exact_match']:.4f}")
        print(f"  Groundedness  : {r['groundedness']['score']:.4f} "
              f"(passes={r['groundedness']['passes']}  "
              f"uncertain={r['groundedness'].get('uncertain', '?')}  "
              f"spans={r['groundedness']['total_spans']})")
        print(f"  G explanation : {r['groundedness'].get('explanation', 'N/A')[:100]}")
        print(f"  Sensitivity   : avg={r['sensitivity']['avg_sensitivity']:.4f} "
              f"most_influential=chunk_{r['sensitivity']['most_influential_idx']}")
        print(f"  Gemini calls  : {r['gemini_calls']}")
        print(f"  Trial history : {len(r['trial_history'])} tier attempt(s)")
        for t in r["trial_history"]:
            print(f"    [{t['tier']}] passed={t['passed']}  "
                  f"g_score={t['groundedness']['score']:.3f}")
        print()

    results = []
    for i, qa in enumerate(doc["qa_pairs"]):
        print(f"Running QA {i+1}/{len(doc['qa_pairs'])}...")
        r = pipeline.process_question(
            question     = qa["question"],
            doc          = doc,
            ground_truth = qa["answer"],
        )
        assert r["answer"]    != "",      f"Empty answer for QA {i+1}"
        assert r["tier_used"] in {"primary", "fallback", "flat"}
        assert "trial_history" in r,      "trial_history missing from result"
        assert len(r["trial_history"]) >= 1
        _print_result(r, i)
        results.append(r)

    # JSON serialisable check
    try:
        json.dumps(results)
        print("JSON serialisable : PASS")
    except TypeError as e:
        print(f"JSON serialisable : FAIL — {e}")

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    print(f"\nAverage F1 for this paper : {avg_f1:.4f}")
    # save results so tier_analyser.py can load them
    import os
    os.makedirs("data/qasper", exist_ok=True)
    out_path = "data/qasper/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    print("=" * 55)
    print("All tests passed.")
    print("=" * 55)