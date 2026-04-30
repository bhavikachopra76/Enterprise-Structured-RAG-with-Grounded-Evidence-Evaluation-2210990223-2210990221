"""
run_experiment.py
Full experiment runner for HierRAG — processes all papers in the dataset.

This is the main script you run for the actual experiment (not hierrag_pipeline.py,
which is for single-paper testing).

Features:
    - Processes every paper and every QA pair in prepared_data.json
    - Saves results incrementally after each paper — crash-safe, so you don't
      lose progress if the process dies halfway through
    - Skips papers that already have a results file (resume from where you left off)
    - Prints a running summary after each paper (F1, groundedness, tier usage)
    - Prints final aggregate stats at the end
    - Logs Gemini API usage every 50 calls for quota monitoring

Usage:
    python run_experiment.py

Output:
    data/qasper/results/          ← one JSON file per paper (incremental)
    data/qasper/results_all.json  ← combined file for tier_analyser.py
"""

import json
import os
import traceback
from tqdm import tqdm
from hierrag_pipeline import HierRAG
from sensitivity_tester import compute_f1
from gemini_client import print_key_status, KeysExhaustedError

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH       = "data/qasper/prepared_data.json"
RESULTS_DIR     = "data/qasper/results"
RESULTS_ALL     = "data/qasper/results_all.json"
PRINT_KEY_EVERY = 50   # print Gemini usage stats every N calls


def already_done(doc_id: str) -> bool:
    """Check if this paper already has a results file on disk."""
    return os.path.exists(os.path.join(RESULTS_DIR, f"{doc_id}.json"))


def load_paper_results(doc_id: str) -> list:
    """Load existing results for a previously processed paper."""
    path = os.path.join(RESULTS_DIR, f"{doc_id}.json")
    with open(path) as f:
        return json.load(f)


def save_paper_results(doc_id: str, results: list):
    """Save results for one paper immediately after processing.

    Writing after each paper (not at the end) makes the experiment crash-safe.
    """
    path = os.path.join(RESULTS_DIR, f"{doc_id}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def combine_all_results() -> list:
    """Combine all per-paper result files into one flat list for analysis."""
    all_results = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                all_results.extend(json.load(f))
    return all_results


def print_paper_summary(doc_title: str, results: list):
    """Print a quick summary after each paper finishes processing."""
    if not results:
        return
    avg_f1    = sum(r["f1"]                       for r in results) / len(results)
    avg_g     = sum(r["groundedness"]["score"]     for r in results) / len(results)
    avg_sens  = sum(r["sensitivity"]["avg_sensitivity"] for r in results) / len(results)

    # count how many questions ended up on each tier
    tier_counts = {}
    for r in results:
        tier_counts[r["tier_used"]] = tier_counts.get(r["tier_used"], 0) + 1

    print(f"\n  Paper   : {doc_title[:60]}")
    print(f"  QA pairs: {len(results)}")
    print(f"  Avg F1  : {avg_f1:.4f}  |  Avg groundedness: {avg_g:.4f}  |  Avg sensitivity: {avg_sens:.4f}")
    print(f"  Tiers   : {tier_counts}")


def print_final_summary(all_results: list):
    """Print aggregate stats across the whole dataset after all papers are done."""
    if not all_results:
        return

    print("\n" + "=" * 60)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 60)

    total_qa  = len(all_results)
    avg_f1    = sum(r["f1"]                            for r in all_results) / total_qa
    avg_g     = sum(r["groundedness"]["score"]          for r in all_results) / total_qa
    avg_sens  = sum(r["sensitivity"]["avg_sensitivity"] for r in all_results) / total_qa
    avg_calls = sum(r["gemini_calls"]                  for r in all_results) / total_qa

    uncertain_count = sum(
        1 for r in all_results if r["groundedness"].get("uncertain", False)
    )

    # tier usage breakdown
    tier_counts = {}
    for r in all_results:
        t = r["tier_used"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    # average F1 per tier — helps assess whether fallback actually helps
    tier_f1s = {}
    for r in all_results:
        t = r["tier_used"]
        tier_f1s.setdefault(t, []).append(r["f1"])
    avg_f1_per_tier = {t: sum(v)/len(v) for t, v in tier_f1s.items()}

    print(f"  Total QA pairs processed : {total_qa}")
    print(f"  Average F1               : {avg_f1:.4f}")
    print(f"  Average groundedness     : {avg_g:.4f}")
    print(f"  Average sensitivity      : {avg_sens:.4f}")
    print(f"  Avg Gemini calls/QA      : {avg_calls:.1f}")
    print(f"  Uncertain groundedness   : {uncertain_count} ({100*uncertain_count/total_qa:.1f}%)")
    print(f"\n  Tier usage:")
    for tier, count in sorted(tier_counts.items()):
        pct = 100 * count / total_qa
        print(f"    {tier:<10} : {count:>4}  ({pct:.1f}%)  avg_F1={avg_f1_per_tier[tier]:.4f}")
    print("=" * 60)


# ── main ──────────────────────────────────────────────────────────────────────
def run():
    """Main experiment loop. Processes all papers, saves results incrementally."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # load the prepared dataset
    print(f"Loading dataset from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        docs = json.load(f)
    print(f"  {len(docs)} papers loaded\n")

    # split into done vs pending using the on-disk result files
    done    = [d for d in docs if     already_done(d["doc_id"])]
    pending = [d for d in docs if not already_done(d["doc_id"])]
    print(f"  Already completed : {len(done)} papers")
    print(f"  Remaining         : {len(pending)} papers\n")

    if not pending:
        print("All papers already processed. Combining results...")
        all_results = combine_all_results()
        with open(RESULTS_ALL, "w") as f:
            json.dump(all_results, f, indent=2)
        print_final_summary(all_results)
        return

    pipeline     = HierRAG()
    total_calls  = 0
    failed_papers = []

    for doc in tqdm(pending, desc="Papers", unit="paper"):
        doc_id    = doc["doc_id"]
        qa_pairs  = doc["qa_pairs"]
        results   = []

        print(f"\nProcessing: {doc['title'][:60]}")
        print(f"  QA pairs: {len(qa_pairs)}")

        for qa in qa_pairs:
            try:
                r = pipeline.process_question(
                    question     = qa["question"],
                    doc          = doc,
                    ground_truth = qa["answer"],
                )
                results.append(r)
                total_calls += r["gemini_calls"]

                # log Gemini usage periodically
                if total_calls % PRINT_KEY_EVERY < r["gemini_calls"]:
                    print_key_status()

            except KeysExhaustedError as e:
                # all API quota exhausted — save partial progress and stop
                print(f"\n  [runner] ALL KEYS EXHAUSTED: {e}")
                print(f"  Saving partial results for {doc_id}...")
                if results:
                    save_paper_results(doc_id, results)
                print("  Stopping experiment. Re-run to resume from next paper.")

                all_results = combine_all_results()
                with open(RESULTS_ALL, "w") as f:
                    json.dump(all_results, f, indent=2)
                print_final_summary(all_results)
                return

            except Exception as e:
                # single QA pair failed — log it and keep going
                print(f"  [runner] QA failed: '{qa['question'][:60]}': {e}")
                traceback.print_exc()
                continue

        # save this paper immediately — don't wait for the full run to finish
        if results:
            save_paper_results(doc_id, results)
            print_paper_summary(doc["title"], results)
        else:
            failed_papers.append(doc_id)
            print(f"  [runner] no results for {doc_id} — skipping save")

    # all papers done — combine per-paper files into one for tier_analyser.py
    print("\nAll papers processed. Combining results...")
    all_results = combine_all_results()
    with open(RESULTS_ALL, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to {RESULTS_ALL}")

    if failed_papers:
        print(f"\nFailed papers ({len(failed_papers)}): {failed_papers}")

    print_final_summary(all_results)


if __name__ == "__main__":
    run()