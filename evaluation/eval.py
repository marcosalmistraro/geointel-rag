"""
RAG evaluation script — keyword recall over 10 test questions.

Metric: for each question, keyword_recall = keywords_found / total_keywords.
A keyword is "found" if it appears (case-insensitive) anywhere in the answer.

Pass threshold: recall >= 0.5 (at least half the expected keywords present).

Usage:
    python -m evaluation.eval

Results are printed to stdout and logged to MLflow under the
"geointel-eval" experiment so runs can be compared over time.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field

import mlflow

from config import settings
from rag.chain import RAGChain

logging.basicConfig(level=logging.WARNING)

PASS_THRESHOLD = 0.5

TEST_CASES = [
    {
        "question": "How many people were killed in the earthquake?",
        "keywords": ["killed", "dead", "deaths", "casualties", "people"],
    },
    {
        "question": "What was the humanitarian situation in Hatay?",
        "keywords": ["hatay", "shelter", "displaced", "damage", "buildings"],
    },
    {
        "question": "How many people were displaced by the earthquake?",
        "keywords": ["displaced", "people", "shelter", "million"],
    },
    {
        "question": "What search and rescue operations were conducted?",
        "keywords": ["rescue", "search", "teams", "survivors"],
    },
    {
        "question": "What was the food security situation in affected areas?",
        "keywords": ["food", "aid", "distribution", "assistance"],
    },
    {
        "question": "How did the earthquake affect Syrian refugees in Turkey?",
        "keywords": ["refugees", "syrian", "turkey", "displaced"],
    },
    {
        "question": "What was the scale of building destruction?",
        "keywords": ["buildings", "destroyed", "collapsed", "damage"],
    },
    {
        "question": "What international assistance was mobilised after the earthquake?",
        "keywords": ["international", "aid", "teams", "support"],
    },
    {
        "question": "What was the situation in Kahramanmaras after the earthquake?",
        "keywords": ["kahramanmaras", "earthquake", "damage", "affected"],
    },
    {
        "question": "What were the health and medical needs after the earthquake?",
        "keywords": ["health", "medical", "hospital", "injuries"],
    },
]


@dataclass
class EvalResult:
    question: str
    answer: str
    keywords: list[str]
    found: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def recall(self) -> float:
        return len(self.found) / len(self.keywords) if self.keywords else 0.0

    @property
    def passed(self) -> bool:
        return self.recall >= PASS_THRESHOLD


def run_eval(model_id: str | None = None) -> list[EvalResult]:
    chain = RAGChain(model_id=model_id) if model_id else RAGChain()
    results: list[EvalResult] = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"  [{i:02d}/{len(TEST_CASES)}] {case['question'][:70]}…", end=" ", flush=True)
        t0 = time.perf_counter()
        raw = chain.run(case["question"])
        latency_ms = (time.perf_counter() - t0) * 1000

        answer_lower = raw["answer"].lower()
        found = [kw for kw in case["keywords"] if kw in answer_lower]

        result = EvalResult(
            question=case["question"],
            answer=raw["answer"],
            keywords=case["keywords"],
            found=found,
            latency_ms=round(latency_ms, 1),
        )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}  recall={result.recall:.2f}  ({latency_ms:.0f} ms)")

    return results


def log_to_mlflow(results: list[EvalResult], model_id: str) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("geointel-eval")

    mean_recall = sum(r.recall for r in results) / len(results)
    pass_rate = sum(r.passed for r in results) / len(results)
    mean_latency = sum(r.latency_ms for r in results) / len(results)

    with mlflow.start_run(run_name=f"eval-{model_id}"):
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("n_questions", len(results))
        mlflow.log_param("pass_threshold", PASS_THRESHOLD)
        mlflow.log_metrics({
            "mean_recall": round(mean_recall, 4),
            "pass_rate": round(pass_rate, 4),
            "mean_latency_ms": round(mean_latency, 1),
        })
        # Log per-question recall as individual metrics
        for i, r in enumerate(results):
            mlflow.log_metric("recall", r.recall, step=i)
        # Full results as artifact
        lines = ["question,recall,passed,latency_ms,found_keywords"]
        for r in results:
            lines.append(
                f'"{r.question}",{r.recall:.4f},{r.passed},'
                f'{r.latency_ms},"{"|".join(r.found)}"'
            )
        mlflow.log_text("\n".join(lines), "eval_results.csv")


def print_summary(results: list[EvalResult], model_id: str) -> None:
    mean_recall = sum(r.recall for r in results) / len(results)
    passed = sum(r.passed for r in results)

    print()
    print("─" * 72)
    print(f"Model:       {model_id}")
    print(f"Questions:   {len(results)}")
    print(f"Passed:      {passed}/{len(results)}  (threshold ≥ {PASS_THRESHOLD})")
    print(f"Mean recall: {mean_recall:.2%}")
    print(f"Mean latency:{sum(r.latency_ms for r in results) / len(results):.0f} ms")
    print("─" * 72)
    print()
    print(f"{'#':<3} {'Recall':<8} {'Pass':<6} Question")
    print(f"{'─'*3} {'─'*8} {'─'*6} {'─'*52}")
    for i, r in enumerate(results, 1):
        tick = "✓" if r.passed else "✗"
        q = r.question[:52]
        print(f"{i:<3} {r.recall:<8.2f} {tick:<6} {q}")
    print()


if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "llama-3.1-8b-instant"

    print(f"\nGeoIntel RAG — Evaluation")
    print(f"Model: {model_id}")
    print(f"Running {len(TEST_CASES)} questions…\n")

    results = run_eval(model_id=model_id)
    print_summary(results, model_id)

    print("Logging to MLflow…", end=" ")
    try:
        log_to_mlflow(results, model_id)
        print("done.")
        print(f"Run `mlflow ui` to browse results.\n")
    except Exception as e:
        print(f"failed ({e})")
