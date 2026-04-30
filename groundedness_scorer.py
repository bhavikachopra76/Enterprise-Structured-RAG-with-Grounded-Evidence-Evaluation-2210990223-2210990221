"""
groundedness_scorer.py
Two-call LLM-based groundedness scoring for HierRAG.

Why two separate LLM calls?
    The LLM that generated the answer can't reliably audit itself in the same
    pass — it anchors on what it just said and selectively pulls supporting
    evidence (self-confirmation bias). By using a SEPARATE verification call,
    the model acts as an impartial auditor, not a defender.

    This follows the same separation used in FActScore (Min et al., 2023)
    and RAGAS faithfulness scoring.

What the verification call does:
    Given: question + context chunks + generated answer
    Returns:
        context_spans  — key verifiable facts in the context
        answer_spans   — verifiable claims made in the answer
        matched        — answer_spans that are supported by context_spans
        unmatched      — answer_spans with no support (potential hallucinations)
        score          — len(matched) / len(answer_spans)
        explanation    — one sentence explaining why the answer passes or fails

Fuzzy matching:
    Exact set intersection misses paraphrases ("AdaGrad optimizer" vs "AdaGrad").
    A fuzzy match with SequenceMatcher ratio >= 0.8 catches these without being
    too permissive. Applied on top of the LLM's own matching to fix misses.

Scoring threshold:
    >= 0.5 → passes (acceptable groundedness)
    < 0.5  → fails  (too many ungrounded claims, triggers tier backtrack)

    Special cases:
        answer is "Not found in context" → score 0.0, fails
        no answer_spans extracted        → score 1.0 but uncertain=True,
                                           so caller knows it was unverifiable
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from gemini_client import call_gemini_json, sleep_between_calls

GROUNDEDNESS_THRESHOLD  = 0.5
FUZZY_MATCH_THRESHOLD   = 0.8   # SequenceMatcher ratio cutoff for span matching

# ── verification prompt ────────────────────────────────────────────────────────
_VERIFY_PROMPT = """You are auditing a generated answer for factual grounding.

You are given a question, the context passages that were retrieved, and a 
generated answer. Your job is to identify which claims in the answer are 
supported by the context and which are not.

Question: {question}

Context passages:
{context}

Generated answer: {answer}

Reply with JSON only, no markdown fences, no other text:
{{
    "context_spans": ["key verifiable fact 1", "key verifiable fact 2", ...],
    "answer_spans":  ["verifiable claim from answer 1", ...],
    "matched":       ["claims supported by context", ...],
    "unmatched":     ["claims NOT found in context", ...],
    "explanation":   "one sentence: why this answer passes or fails groundedness"
}}

Rules:
- context_spans: factual spans worth verifying — names, numbers, model names,
  dataset names, metric values, technical terms. Be exhaustive (5-15 items).
- answer_spans: only verifiable claims the answer makes — skip filler phrases
  like "they build a classifier that". Focus on specific facts.
- matched: answer_spans that are supported by context_spans (allow paraphrasing
  — "AdaGrad optimizer" matches "AdaGrad")
- unmatched: answer_spans with NO support in context — these are hallucinations
  or unsupported claims
- if answer says "Not found in context", all lists must be empty
- explanation: be specific — name the unmatched spans if any"""


# ── not-found detector ─────────────────────────────────────────────────────────
_NOT_FOUND_PHRASES = [
    "not found in context",
    "not mentioned in the context",
    "context does not contain",
    "cannot be found",
    "no information",
    "not provided in",
]

def _is_not_found(answer: str) -> bool:
    """Check if the answer is basically saying 'I couldn't find anything.'"""
    a = answer.lower()
    return any(p in a for p in _NOT_FOUND_PHRASES)


# ── fuzzy span matching ────────────────────────────────────────────────────────
def _fuzzy_match(span: str, candidates: list, threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """Return True if `span` fuzzy-matches any candidate at >= threshold ratio.

    This catches paraphrases the LLM's own matching might miss,
    e.g. "AdaGrad optimizer" vs "AdaGrad" (ratio ~0.82).
    """
    span_norm = span.lower().strip()
    for c in candidates:
        ratio = SequenceMatcher(None, span_norm, c.lower().strip()).ratio()
        if ratio >= threshold:
            return True
    return False


# ── result dataclass ───────────────────────────────────────────────────────────
@dataclass
class GroundednessResult:
    """Holds the outcome of a groundedness check.

    Key fields:
        score:     fraction of answer_spans that are grounded (0.0 to 1.0)
        passes:    True if score >= GROUNDEDNESS_THRESHOLD
        uncertain: True if no verifiable claims could be extracted at all
    """
    score          : float
    total_spans    : int
    matched_spans  : int
    matched_list   : list = field(default_factory=list)
    unmatched_list : list = field(default_factory=list)
    context_spans  : list = field(default_factory=list)
    explanation    : str  = ""
    passes         : bool = False
    uncertain      : bool = False   # True when no answer_spans were extractable

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON output."""
        return {
            "score"         : round(self.score, 4),
            "total_spans"   : self.total_spans,
            "matched_spans" : self.matched_spans,
            "matched_list"  : self.matched_list,
            "unmatched_list": self.unmatched_list,
            "context_spans" : self.context_spans,
            "explanation"   : self.explanation,
            "passes"        : self.passes,
            "uncertain"     : self.uncertain,
        }


# ── main scorer ────────────────────────────────────────────────────────────────
class GroundednessScorer:
    """Two-call LLM groundedness scorer.

    Call 1 (in the pipeline): generates the answer — happens before score() is called.
    Call 2 (here in score()): verifies the answer — separate auditor role call.

    The pipeline calls score() after every generation attempt to decide
    whether to accept the answer or fall back to the next tier.
    """

    def score(
        self,
        answer : str,
        chunks : list[str],
        question: str = "",
    ) -> GroundednessResult:
        """Verify how many claims in `answer` are supported by `chunks`.

        Args:
            answer:   the generated answer string (already produced by pipeline)
            chunks:   list of retrieved context chunk strings
            question: the original question (helps the auditor LLM understand relevance)

        Returns:
            GroundednessResult with score, matched/unmatched spans, and pass/fail.
        """
        # fast path — answer explicitly says nothing was found
        if _is_not_found(answer):
            return GroundednessResult(
                score       = 0.0,
                total_spans = 0,
                matched_spans = 0,
                explanation = "Answer explicitly states the context contains no answer.",
                passes      = False,
                uncertain   = False,
            )

        # join chunks with separators for the verification prompt
        context = "\n\n---\n\n".join(chunks)

        prompt = _VERIFY_PROMPT.format(
            question = question or "Not provided",
            context  = context,
            answer   = answer,
        )

        try:
            result = call_gemini_json(prompt, temperature=0.0)
            sleep_between_calls(2.0)
            # debug logging — shows what the auditor extracted
            print(f"  [groundedness] context_spans={result.get('context_spans',[])} ")
            print(f"  [groundedness] answer_spans={result.get('answer_spans',[])} ")
        except Exception as e:
            # if verification call fails, we let the answer through with uncertain=True
            # so the pipeline isn't blocked — but it's clearly flagged
            print(f"  [groundedness] verification call failed: {e} — marking uncertain")
            return GroundednessResult(
                score       = 1.0,
                total_spans = 0,
                matched_spans = 0,
                explanation = f"Verification call failed: {e}",
                passes      = True,
                uncertain   = True,
            )

        # parse the auditor's JSON response
        context_spans  = result.get("context_spans", [])
        answer_spans   = result.get("answer_spans",  [])
        llm_matched    = result.get("matched",        [])
        llm_unmatched  = result.get("unmatched",      [])
        explanation    = result.get("explanation",    "")

        # edge case — auditor found no verifiable claims in the answer
        if not answer_spans:
            return GroundednessResult(
                score         = 1.0,
                total_spans   = 0,
                matched_spans = 0,
                context_spans = context_spans,
                explanation   = explanation or "No verifiable claims found in answer.",
                passes        = True,
                uncertain     = True,   # flagged: we didn't actually verify anything
            )

        # re-verify matched/unmatched with fuzzy matching on top of LLM output.
        # the LLM sometimes misses paraphrase matches — fuzzy catches them.
        verified_matched   = []
        verified_unmatched = []

        for span in answer_spans:
            # check LLM's matched list first, then try fuzzy against context_spans
            if span in llm_matched or _fuzzy_match(span, context_spans):
                verified_matched.append(span)
            else:
                verified_unmatched.append(span)

        score  = len(verified_matched) / len(answer_spans)
        passes = score >= GROUNDEDNESS_THRESHOLD

        return GroundednessResult(
            score         = round(score, 4),
            total_spans   = len(answer_spans),
            matched_spans = len(verified_matched),
            matched_list  = verified_matched,
            unmatched_list= verified_unmatched,
            context_spans = context_spans,
            explanation   = explanation,
            passes        = passes,
            uncertain     = False,
        )


# ── TEST ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("groundedness_scorer.py — running tests")
    print("=" * 55)

    scorer = GroundednessScorer()

    # test 1 — perfectly grounded answer
    print("\nTest 1: perfectly grounded answer")
    answer  = "The authors used the OntoNotes 5.0 dataset with 2802 training documents."
    context = ["The model was evaluated on OntoNotes 5.0 benchmark with 2802 training documents in the CoNLL format."]
    result  = scorer.score(answer, context, question="What dataset was used?")
    print(f"  score       : {result.score}")
    print(f"  matched     : {result.matched_list}")
    print(f"  unmatched   : {result.unmatched_list}")
    print(f"  explanation : {result.explanation}")
    print(f"  uncertain   : {result.uncertain}")
    print(f"  passes      : {result.passes}")
    assert result.passes, f"Should pass — all spans grounded. Score={result.score}"
    print("  PASS\n")

    # test 2 — partially grounded (hallucinated entity)
    print("Test 2: partially grounded — hallucinated entity")
    answer  = "The model achieved 73.9% F1 on OntoNotes and was trained in New York."
    context = ["The model achieved 73.9% F1 score on the OntoNotes benchmark dataset."]
    result  = scorer.score(answer, context, question="What were the results?")
    print(f"  score       : {result.score}")
    print(f"  unmatched   : {result.unmatched_list}")
    print(f"  explanation : {result.explanation}")
    assert result.score < 1.0, f"Score should be < 1.0 — New York is hallucinated"
    print("  PASS\n")

    # test 3 — fully hallucinated
    print("Test 3: fully hallucinated answer")
    answer  = "The model was trained in Berlin by the Google team in 2021."
    context = ["We trained our model using the standard NLP benchmark dataset."]
    result  = scorer.score(answer, context, question="Who trained the model?")
    print(f"  score       : {result.score}")
    print(f"  unmatched   : {result.unmatched_list}")
    print(f"  explanation : {result.explanation}")
    assert not result.passes, f"Should fail — nothing grounded. Score={result.score}"
    print("  PASS\n")

    # test 4 — not found answer
    print("Test 4: not found in context")
    answer  = "Not found in context."
    context = ["Some unrelated text about something else entirely."]
    result  = scorer.score(answer, context, question="What is the answer?")
    print(f"  score   : {result.score}")
    print(f"  passes  : {result.passes}")
    assert result.score == 0.0
    assert not result.passes
    print("  PASS\n")

    # test 5 — the QA1 case from actual results (was silently passing before)
    print("Test 5: QA1 case — topics pulled from Reddit (was silently passing with spaCy)")
    answer  = "The topics mentioned are abortion, climate change, cooking, politics, religion, photography, homebrewing, socially awkward messages, and internet or social media fights."
    context = [
        "A natural starting point for analyzing dogmatism on Reddit is to examine how it characterizes the site's sub-communities. For example, we might expect to see that subreddits oriented around topics such as abortion or climate change are more dogmatic, and subreddits about cooking are less so.",
        "The subreddit with the highest average dogmatism level, cringepics, is a place to make fun of socially awkward messages. Similarly, SubredditDrama is a community where people come to talk about fights on the internet or social media.",
    ]
    result  = scorer.score(answer, context, question="What are the topics pulled from Reddit?")
    print(f"  score         : {result.score}")
    print(f"  total_spans   : {result.total_spans}")
    print(f"  matched       : {result.matched_list}")
    print(f"  unmatched     : {result.unmatched_list}")
    print(f"  explanation   : {result.explanation}")
    print(f"  uncertain     : {result.uncertain}")
    print(f"  passes        : {result.passes}")
    # should now have actual spans extracted — not uncertain
    assert result.total_spans > 0, \
        f"Should have extracted spans this time (was 0 with spaCy). Got {result.total_spans}"
    print("  PASS — no longer silently passing with total_spans=0\n")

    # test 6 — to_dict format
    print("Test 6: to_dict format")
    d = result.to_dict()
    for key in ["score", "total_spans", "matched_spans", "matched_list",
                "unmatched_list", "context_spans", "explanation", "passes", "uncertain"]:
        assert key in d, f"Missing key: {key}"
    print(f"  keys: {list(d.keys())}")
    print("  PASS\n")

    print("=" * 55)
    print("All tests passed.")
    print("=" * 55)