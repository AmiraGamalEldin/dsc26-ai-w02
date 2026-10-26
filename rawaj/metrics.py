"""Decision metrics for a mule-risk score.

You write this file. Keep the signatures exactly as they are: the grader calls them by name.

Rules for every function:
- y_true and scores are lists or 1-D arrays. Raise ValueError if the lengths differ, the
  input is empty, a label is not 0 or 1, or a score is NaN.
- A case is flagged when score >= threshold.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
"""
from __future__ import annotations
import math

def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return {"tp": int, "fp": int, "tn": int, "fn": int} at `threshold`."""
    
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")

    if any(y not in (0, 1) for y in y_true):
        raise ValueError("Labels must be 0 or 1")

    if any(score != score for score in scores):
        raise ValueError("Scores must not contain NaN")

    tp = fp = tn = fn = 0

    for actual, score in zip(y_true, scores):
        flagged = score >= threshold

        if actual == 1 and flagged:
            tp += 1
        elif actual == 0 and flagged:
            fp += 1
        elif actual == 0 and not flagged:
            tn += 1
        else:
            fn += 1

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}



def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Return (precision, recall) at `threshold`."""

    counts = confusion_at(y_true, scores, threshold)

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall



def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores."""

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")

    if any(y not in (0, 1) for y in y_true):
        raise ValueError("Labels must be 0 or 1")

    if any(score != score for score in scores):
        raise ValueError("Scores must not contain NaN")

    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, len(y_true))

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    top_indices = ranked_indices[:k]
    positives = sum(y_true[i] == 1 for i in top_indices)

    return positives / k




def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """fp * cost_fp + fn * cost_fn at `threshold`."""

    counts = confusion_at(y_true, scores, threshold)

    return counts["fp"] * cost_fp + counts["fn"] * cost_fn



def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest expected cost.

    On a tie in cost, return the highest threshold.
    """

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")
    if any(y not in (0, 1) for y in y_true):
        raise ValueError("Labels must be 0 or 1")
    if any(score != score for score in scores):
        raise ValueError("Scores must not contain NaN")

    candidates = sorted(set(scores)) + [math.inf]

    return min(
        candidates,
        key=lambda t: (
            expected_cost(y_true, scores, t, cost_fp=cost_fp, cost_fn=cost_fn),
            -t,
        ),
    )




def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest distinct score whose precision is >= min_precision, or None if none reaches it."""

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")

    if any(y not in (0, 1) for y in y_true):
        raise ValueError("Labels must be 0 or 1")

    if any(score != score for score in scores):
        raise ValueError("Scores must not contain NaN")

    candidates = sorted(set(scores))

    for threshold in candidates:
        precision, recall = precision_recall_at(y_true, scores, threshold)

        if precision >= min_precision:
            return threshold

    return None

