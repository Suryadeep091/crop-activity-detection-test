import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_exclusions(path: Optional[Path]) -> Set[str]:
    if not path:
        return set()
    data = load_json(path)
    if isinstance(data, list):
        return {str(item) for item in data}
    if isinstance(data, dict):
        return {str(item) for item in data.get("exclude_task_ids", [])}
    return set()


def infer_truth(task_id: str) -> str:
    if "(Crop)" in task_id:
        return "Active"
    if "(No-Crop)" in task_id:
        return "Inactive"
    return "Unknown"


def infer_state(task_id: str) -> str:
    match = re.match(r"([A-Za-z]+)", task_id.strip())
    return match.group(1).upper() if match else "UNKNOWN"


def normalize_prediction(entry: Dict) -> str:
    decision_label = str(entry.get("decision_label") or "").strip().title()
    if decision_label == "Review":
        return "Review"
    return "Active" if bool(entry.get("is_active", False)) else "Inactive"


def classify_outcome(truth: str, prediction: str) -> str:
    if truth == "Unknown":
        return "Unknown"
    if prediction == "Review":
        return "Review"
    if truth == prediction:
        return "TP" if truth == "Active" else "TN"
    if truth == "Inactive" and prediction == "Active":
        return "FP"
    if truth == "Active" and prediction == "Inactive":
        return "FN"
    return "Unknown"


def percent(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100, 2) if denominator else 0.0


def summarize(entries: Iterable[Dict], exclusions: Set[str]) -> Dict:
    kept: List[Dict] = []
    skipped = 0
    for entry in entries:
        task_id = str(entry.get("task_id", ""))
        if task_id in exclusions:
            skipped += 1
            continue
        truth = infer_truth(task_id)
        prediction = normalize_prediction(entry)
        outcome = classify_outcome(truth, prediction)
        kept.append({
            **entry,
            "truth": truth,
            "prediction": prediction,
            "outcome": outcome,
            "state": infer_state(task_id),
        })

    outcome_counts = Counter(item["outcome"] for item in kept)
    decisive = [item for item in kept if item["prediction"] != "Review" and item["truth"] != "Unknown"]
    correct = sum(1 for item in decisive if item["truth"] == item["prediction"])

    by_state = defaultdict(Counter)
    by_error_type = defaultdict(list)
    for item in kept:
        by_state[item["state"]][item["outcome"]] += 1
        if item["outcome"] in {"FP", "FN", "Review"}:
            by_error_type[item["outcome"]].append({
                "task_id": item.get("task_id"),
                "truth": item["truth"],
                "prediction": item["prediction"],
                "activity_score": item.get("activity_score"),
                "decision_reason": item.get("decision_reason"),
                "data_quality": item.get("data_quality", {}),
            })

    return {
        "total_input": len(list(entries)) if not isinstance(entries, list) else len(entries),
        "total_evaluated": len(kept),
        "excluded": skipped,
        "decisive_accuracy_percent": percent(correct, len(decisive)),
        "review_rate_percent": percent(outcome_counts["Review"], len(kept)),
        "outcome_counts": dict(outcome_counts),
        "by_state": {state: dict(counts) for state, counts in sorted(by_state.items())},
        "by_error_type": dict(by_error_type),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate AdvaRisk decision-policy results by state and error type without parcel-specific tuning."
    )
    parser.add_argument("results_json", type=Path, help="Replay/live results JSON file.")
    parser.add_argument("--exclude", type=Path, help="Optional JSON list or {'exclude_task_ids': [...]} file.")
    parser.add_argument("--output", type=Path, help="Optional path to write validation summary JSON.")
    args = parser.parse_args()

    entries = load_json(args.results_json)
    if not isinstance(entries, list):
        raise ValueError("Expected results_json to contain a list of API result objects.")

    summary = summarize(entries, load_exclusions(args.exclude))
    rendered = json.dumps(summary, indent=2)
    print(rendered)

    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
