"""
tier_analyser.py
Post-hoc analysis of tier transitions in HierRAG results.

After the pipeline has finished processing papers and saved results to JSON,
run this analyser to understand WHY tier transitions happened — i.e., why
the primary retrieval failed and whether the fallback actually helped.

What it does:
    1. Loads a results JSON produced by run_experiment.py or hierrag_pipeline.py
    2. For every question where tier 2 or tier 3 was used (primary failed),
       asks Gemini to explain:
           - Why did tier 1's answer fail groundedness?
           - What was wrong or missing in the primary chunks?
           - Did the fallback tier produce a better answer?
    3. Computes aggregate stats:
           - How often each tier was used (primary vs fallback vs flat)
           - Average F1 / groundedness score per tier
           - Correlation between groundedness score and tier used
    4. Identifies "recovery examples" — questions where the fallback tier
       actually improved F1 over the primary attempt
    5. Writes a full analysis JSON and prints a human-readable summary

Usage:
    python tier_analyser.py --results data/qasper/results_all.json
    python tier_analyser.py --results data/qasper/results_all.json --output analysis.json

    Or import and call analyse() directly from another script.
"""

import json
import argparse
import numpy as np
from dataclasses import dataclass, field
from gemini_client import call_gemini_json, sleep_between_calls


# ── LLM explanation prompt ────────────────────────────────────────────────────
_EXPLAIN_PROMPT = """You are analysing why a RAG (Retrieval-Augmented Generation) 
system had to fall back from its primary retrieved context to a secondary one.

You are given:
- The question asked
- The primary context chunks that were retrieved first
- The answer generated from the primary context
- The groundedness evaluation of that answer (score, matched spans, unmatched spans)
- The fallback context that was used instead (if any)
- The final answer produced

Your job: explain concisely why the primary retrieval failed groundedness, and 
whether the fallback actually helped.

Question: {question}

--- PRIMARY TIER ---
Context chunks used:
{primary_chunks}

Answer generated:
{primary_answer}

Groundedness result:
  Score         : {primary_g_score}
  Matched spans : {primary_matched}
  Unmatched spans (possible hallucinations): {primary_unmatched}

--- FALLBACK TIER ({fallback_tier}) ---
Answer produced:
{fallback_answer}

Groundedness of final answer:
  Score  : {final_g_score}
  Passes : {final_passes}

Reply with JSON only, no markdown fences:
{{
    "why_primary_failed": "1-2 sentences: what was missing or wrong in the primary context",
    "hallucination_pattern": "brief description of the ungrounded spans — were they fabricated details, wrong numbers, wrong names?",
    "did_fallback_help": true or false,
    "why_fallback_helped_or_not": "1-2 sentences explaining whether the fallback recovered the answer",
    "recommendation": "one sentence: what kind of question or context causes this failure pattern"
}}"""


# ── result dataclass ──────────────────────────────────────────────────────────
@dataclass
class TierAnalysis:
    """Holds the complete tier analysis output.

    explanations:      per-question LLM explanations (only for fallback questions)
    tier_counts:       {"primary": N, "fallback": M, "flat": K}
    avg_f1_per_tier:   {"primary": 0.XX, ...}
    avg_g_per_tier:    {"primary": 0.XX, ...}
    recovery_examples: questions where fallback improved F1 over primary
    g_tier_correlation: Pearson correlation between groundedness and tier ordinal
    """
    explanations     : list  = field(default_factory=list)
    tier_counts      : dict  = field(default_factory=dict)
    avg_f1_per_tier  : dict  = field(default_factory=dict)
    avg_g_per_tier   : dict  = field(default_factory=dict)
    recovery_examples: list  = field(default_factory=list)
    g_tier_correlation: float = 0.0

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON output."""
        return {
            "tier_counts"        : self.tier_counts,
            "avg_f1_per_tier"    : self.avg_f1_per_tier,
            "avg_g_per_tier"     : self.avg_g_per_tier,
            "g_tier_correlation" : round(self.g_tier_correlation, 4),
            "recovery_examples"  : self.recovery_examples,
            "explanations"       : self.explanations,
        }


# ── main analyser ─────────────────────────────────────────────────────────────
def analyse(results: list, max_explanations: int = 20) -> TierAnalysis:
    """Analyse a list of pipeline result dicts.

    Args:
        results:          list of dicts from process_question()
        max_explanations: cap on LLM explanation calls to control API cost
                          (fallback questions are analysed in order until cap)

    Returns:
        TierAnalysis dataclass with all stats and explanations.
    """
    analysis = TierAnalysis()

    # ── 1. aggregate stats ────────────────────────────────────────────────────
    tier_f1s = {}
    tier_gs  = {}

    for r in results:
        tier = r["tier_used"]
        tier_f1s.setdefault(tier, []).append(r["f1"])
        tier_gs.setdefault(tier, []).append(r["groundedness"]["score"])

    analysis.tier_counts = {t: len(v) for t, v in tier_f1s.items()}
    analysis.avg_f1_per_tier = {
        t: round(float(np.mean(v)), 4) for t, v in tier_f1s.items()
    }
    analysis.avg_g_per_tier = {
        t: round(float(np.mean(v)), 4) for t, v in tier_gs.items()
    }

    # ── 2. groundedness-tier correlation ──────────────────────────────────────
    # encode tier as ordinal: primary=0, fallback=1, flat=2
    # a negative correlation means higher tier = lower groundedness (expected)
    tier_order = {"primary": 0, "fallback": 1, "flat": 2}
    tier_vals  = [tier_order.get(r["tier_used"], 0) for r in results]
    g_vals     = [r["groundedness"]["score"] for r in results]
    if len(set(tier_vals)) > 1:
        analysis.g_tier_correlation = float(np.corrcoef(g_vals, tier_vals)[0, 1])

    # ── 3. recovery examples — did the fallback actually improve F1? ──────────
    for r in results:
        if r["tier_used"] in ("fallback", "flat") and len(r["trial_history"]) >= 2:
            tier1_attempt = r["trial_history"][0]   # the primary attempt
            from sensitivity_tester import compute_f1
            f1_primary = compute_f1(tier1_attempt["answer"], r["ground_truth"])
            f1_final   = r["f1"]

            if f1_final > f1_primary:
                analysis.recovery_examples.append({
                    "question"      : r["question"],
                    "tier_used"     : r["tier_used"],
                    "f1_primary"    : round(f1_primary, 4),
                    "f1_final"      : round(f1_final,   4),
                    "f1_improvement": round(f1_final - f1_primary, 4),
                    "primary_answer": tier1_attempt["answer"],
                    "final_answer"  : r["answer"],
                })

    # sort by biggest F1 improvement first — most interesting cases on top
    analysis.recovery_examples.sort(key=lambda x: x["f1_improvement"], reverse=True)

    # ── 4. LLM explanations for fallback questions ────────────────────────────
    fallback_results = [
        r for r in results
        if r["tier_used"] in ("fallback", "flat")
        and len(r.get("trial_history", [])) >= 1
    ]

    explain_count = 0
    for r in fallback_results:
        if explain_count >= max_explanations:
            break

        # find the tier 1 attempt in trial history
        tier1 = next(
            (t for t in r["trial_history"] if t["tier"] == "primary"),
            None
        )
        if tier1 is None:
            continue   # no primary attempt recorded — nothing to explain

        # find whichever fallback attempt was used
        fallback_attempt = next(
            (t for t in r["trial_history"] if t["tier"] != "primary"),
            None
        )
        fallback_tier   = fallback_attempt["tier"]  if fallback_attempt else "unknown"
        fallback_answer = fallback_attempt["answer"] if fallback_attempt else "N/A"

        prompt = _EXPLAIN_PROMPT.format(
            question        = r["question"],
            primary_chunks  = "\n---\n".join(tier1["chunks"])[:1500],  # truncate to fit context
            primary_answer  = tier1["answer"],
            primary_g_score = tier1["groundedness"]["score"],
            primary_matched = tier1["groundedness"]["matched_list"],
            primary_unmatched = tier1["groundedness"]["unmatched_list"],
            fallback_tier   = fallback_tier,
            fallback_answer = fallback_answer,
            final_g_score   = r["groundedness"]["score"],
            final_passes    = r["groundedness"]["passes"],
        )

        try:
            explanation = call_gemini_json(prompt, temperature=0.0)
            sleep_between_calls(5.0)

            analysis.explanations.append({
                "question"               : r["question"],
                "tier_used"              : r["tier_used"],
                "primary_g_score"        : tier1["groundedness"]["score"],
                "final_g_score"          : r["groundedness"]["score"],
                "f1"                     : r["f1"],
                **explanation,
            })
            explain_count += 1

        except Exception as e:
            print(f"  [analyser] explanation failed for '{r['question'][:60]}': {e}")
            sleep_between_calls(5.0)

    return analysis


# ── pretty print summary ──────────────────────────────────────────────────────
def print_summary(analysis: TierAnalysis):
    """Print a human-readable summary of the tier analysis to stdout."""
    print("\n" + "=" * 60)
    print("TIER ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n-- Tier usage --")
    total = sum(analysis.tier_counts.values())
    for tier, count in sorted(analysis.tier_counts.items()):
        pct = 100 * count / total if total else 0
        print(f"  {tier:<10} : {count:>4} questions  ({pct:.1f}%)")

    print("\n-- Average F1 per tier --")
    for tier, avg in sorted(analysis.avg_f1_per_tier.items()):
        print(f"  {tier:<10} : {avg:.4f}")

    print("\n-- Average groundedness score per tier --")
    for tier, avg in sorted(analysis.avg_g_per_tier.items()):
        print(f"  {tier:<10} : {avg:.4f}")

    print(f"\n-- Groundedness-tier correlation --")
    print(f"  {analysis.g_tier_correlation:.4f}")
    print(f"  (higher tier = lower groundedness: negative correlation expected)")

    print(f"\n-- Recovery examples (fallback improved F1) --")
    if analysis.recovery_examples:
        for ex in analysis.recovery_examples[:5]:
            print(f"  Q: {ex['question'][:70]}")
            print(f"     tier={ex['tier_used']}  "
                  f"F1: {ex['f1_primary']:.3f} -> {ex['f1_final']:.3f}  "
                  f"(+{ex['f1_improvement']:.3f})")
    else:
        print("  None — fallback did not improve F1 on any question")

    print(f"\n-- LLM explanations generated: {len(analysis.explanations)} --")
    for ex in analysis.explanations[:3]:
        print(f"\n  Q: {ex['question'][:70]}")
        print(f"  Tier: {ex['tier_used']}  |  primary_g={ex['primary_g_score']:.3f}")
        print(f"  Why primary failed : {ex.get('why_primary_failed', 'N/A')}")
        print(f"  Hallucination      : {ex.get('hallucination_pattern', 'N/A')}")
        print(f"  Fallback helped    : {ex.get('did_fallback_help', 'N/A')}")
        print(f"  Recommendation     : {ex.get('recommendation', 'N/A')}")

    print("\n" + "=" * 60)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HierRAG tier transition analyser")
    parser.add_argument(
        "--results", required=True,
        help="Path to results JSON file produced by the pipeline"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write analysis JSON (optional, prints to stdout if omitted)"
    )
    parser.add_argument(
        "--max-explanations", type=int, default=20,
        help="Max number of LLM explanation calls (default: 20)"
    )
    args = parser.parse_args()

    print(f"Loading results from: {args.results}")
    with open(args.results) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results\n")

    print("Running analysis...")
    analysis = analyse(results, max_explanations=args.max_explanations)

    print_summary(analysis)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(analysis.to_dict(), f, indent=2)
        print(f"\nAnalysis saved to: {args.output}")
    else:
        print("\nFull analysis JSON:")
        print(json.dumps(analysis.to_dict(), indent=2))