

# HierRAG — Hierarchical Retrieval-Augmented Generation

### Type : Research Paper
### Bhavika Chopra - 2210990221
### Chaitanya Garg - 2210990233
### Current Status : Prepared, Ready to Submit

A vectorless, LLM-guided retrieval pipeline for question answering over research papers. Instead of using embeddings to find relevant context, HierRAG builds a hierarchical document tree and uses Gemini to *navigate* it — picking the right sections before generating an answer.

Evaluated on the [QASPER](https://allenai.org/data/qasper) dataset (question answering over NLP papers).

---

## How It Works

### The 3-Level Document Tree

Every paper is represented as a tree:

```
Level 0 — Root       (paper title + abstract)
Level 1 — Sections   (one node per section, with a Gemini-generated summary)
Level 2 — Chunks     (sentence-boundary splits of section text, ~150 words each)
```

Summaries are generated in **one batch Gemini call per paper** and cached to disk, so subsequent runs don't cost any API calls.

### The 3-Tier Retrieval Flow

For each question, the pipeline tries up to 3 tiers:

| Tier | Strategy | When |
|------|----------|------|
| **Tier 1 — Primary** | Gemini reads the section map and picks 1-2 sections → generates answer → groundedness check | Always tried first |
| **Tier 2 — Fallback** | If tier 1 fails groundedness → try the backup section Gemini also picked | When tier 1 answer has too many ungrounded claims |
| **Tier 3 — Flat** | If tier 2 also fails → dump all chunks into the prompt | Last resort |

### Groundedness Scoring

A separate Gemini call acts as an **auditor** — it examines the generated answer against the retrieved context and identifies which claims are supported and which aren't. This prevents the model from just hallucinating and calling it a day.

The scorer uses fuzzy matching (SequenceMatcher ≥ 0.8) on top of the LLM's own matching to catch paraphrases like "AdaGrad optimizer" vs "AdaGrad".

**Threshold:** ≥ 0.5 supported claims → pass. Below that → fall back to the next tier.

### Sensitivity Testing

After answering, a leave-one-out analysis measures how much each chunk *actually mattered*:

```
sensitivity(chunk_i) = 1 - cosine_similarity(baseline_answer, answer_without_chunk_i)
```

High sensitivity = that chunk was load-bearing. Low = it was noise. Uses sentence embeddings (`all-MiniLM-L6-v2`) to compare answers, not token-level F1 — this captures semantic shifts regardless of phrasing.

---

## Project Structure

```
├── gemini_client.py          # Vertex AI Gemini wrapper (retry, rate limiting)
├── prepare_dataset.py        # QASPER → universal document format converter
├── hierarchical_index.py     # 3-level tree builder + sentence-boundary chunker
├── branch_selector.py        # LLM-guided section navigation (vectorless)
├── hierrag_pipeline.py       # Main pipeline: ties retrieval → generation → evaluation
├── groundedness_scorer.py    # Two-call LLM groundedness verification
├── sensitivity_tester.py     # Leave-one-out chunk influence analysis
├── run_experiment.py         # Full experiment runner (crash-safe, resumable)
├── tier_analyser.py          # Post-hoc analysis of tier transitions
├── test_url.py               # Quick Vertex AI connectivity diagnostic
├── requirements.txt          # Python dependencies
├── .env                      # GCP credentials (not committed)
└── data/
    └── qasper/
        ├── prepared_data.json    # Converted dataset
        ├── tree_cache/           # Cached document trees (JSON)
        ├── results/              # Per-paper result files
        └── results_all.json      # Combined results for analysis
```

---

## Setup

### Prerequisites

- Python 3.10+
- A Google Cloud project with Vertex AI API enabled
- A service account with the **Vertex AI User** role

### Installation

```bash
# clone the repo
git clone https://github.com/your-username/Hier-Rag.git
cd Hier-Rag

# create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GCP_PROJECT_ID=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=C:/path/to/your/service-account.json
```

The `GOOGLE_APPLICATION_CREDENTIALS` path should point to your GCP service account JSON key file.

### Verify Connectivity

```bash
python test_url.py
```

If this prints a response from Gemini, you're good to go.

---

## Usage

### Step 1: Prepare the Dataset

Downloads QASPER from HuggingFace and converts it to the universal document format:

```bash
python prepare_dataset.py
```

This saves `data/qasper/prepared_data.json` with up to 100 papers and 3 QA pairs each.

### Step 2: Run the Experiment

```bash
python run_experiment.py
```

This processes every paper and QA pair. Results are saved **incrementally** after each paper, so if the process crashes or runs out of quota, you can just re-run it and it picks up where it left off.

Output:
- `data/qasper/results/` — one JSON file per paper
- `data/qasper/results_all.json` — combined file for analysis

### Step 3: Analyse Results

```bash
python tier_analyser.py --results data/qasper/results_all.json --output analysis.json
```

This produces:
- Tier usage breakdown (how often primary vs fallback vs flat was used)
- Average F1 and groundedness per tier
- Recovery examples (where the fallback actually improved the answer)
- LLM-generated explanations for why specific questions triggered fallbacks

---

## Running Individual Tests

Each module has a `__main__` test block. Run any file directly to test it in isolation:

```bash
python gemini_client.py          # tests Vertex AI connectivity + JSON parsing
python hierarchical_index.py     # builds a tree, checks chunking quality
python branch_selector.py        # tests Gemini navigation on a sample paper
python groundedness_scorer.py    # tests grounded/hallucinated/not-found cases
python sensitivity_tester.py     # tests cosine similarity + leave-one-out
python hierrag_pipeline.py       # end-to-end test on one paper
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No embeddings for retrieval** | Gemini reads section summaries and picks sections directly — avoids embedding quality issues and keeps the pipeline simple |
| **Batch summaries (1 call per paper)** | Generating all section summaries in one prompt cuts API costs by ~5-10x |
| **Two-call groundedness** | The generating LLM can't reliably audit itself (self-confirmation bias), so a separate call acts as an independent verifier |
| **Cosine similarity for sensitivity** | F1-delta measures answer quality, not chunk influence. Cosine sim captures whether the *content* shifted regardless of ground truth |
| **temperature=0 everywhere** | Deterministic output means any change in sensitivity testing is from the removed chunk, not sampling variance |
| **Crash-safe incremental saves** | Research experiments are long-running. Saving after each paper means you never lose more than one paper's work |

---

## Metrics

The pipeline reports these metrics for each question:

| Metric | Description |
|--------|-------------|
| `f1` | Token-level F1 against ground truth (after normalization) |
| `exact_match` | 1.0 if normalized prediction == normalized ground truth |
| `groundedness.score` | Fraction of answer claims supported by context (0.0–1.0) |
| `sensitivity.avg_sensitivity` | Mean sensitivity across all chunks (-1 if skipped) |
| `tier_used` | Which retrieval tier produced the final answer |
| `gemini_calls` | Total Gemini API calls used for this question |

---

## License

This project is for research purposes.
