from typing import Dict, Iterable, List, Set


def read_complexes(path: str) -> List[Set[str]]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return [set(line.split()) for line in handle if line.split()]


def neighbor_affinity(first: Set[str], second: Set[str]) -> float:
    if not first or not second:
        return 0.0
    overlap = len(first & second)
    return overlap * overlap / (len(first) * len(second))


def evaluate_complexes(
    predicted: Iterable[Set[str]],
    gold: Iterable[Set[str]],
    threshold: float,
) -> Dict[str, float]:
    predictions = list(predicted)
    references = list(gold)
    matched_predictions = sum(
        any(neighbor_affinity(prediction, reference) >= threshold for reference in references)
        for prediction in predictions
    )
    matched_references = sum(
        any(neighbor_affinity(prediction, reference) >= threshold for prediction in predictions)
        for reference in references
    )
    precision = matched_predictions / len(predictions) if predictions else 0.0
    recall = matched_references / len(references) if references else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    sensitivity_denominator = sum(len(reference) for reference in references)
    sensitivity_numerator = sum(
        max((len(reference & prediction) for prediction in predictions), default=0)
        for reference in references
    )
    sensitivity = sensitivity_numerator / sensitivity_denominator if sensitivity_denominator else 0.0
    ppv_numerator = sum(
        max((len(prediction & reference) for reference in references), default=0)
        for prediction in predictions
    )
    ppv_denominator = sum(
        len(prediction & reference)
        for prediction in predictions
        for reference in references
    )
    ppv = ppv_numerator / ppv_denominator if ppv_denominator else 0.0
    accuracy = (sensitivity * ppv) ** 0.5
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sn": sensitivity,
        "ppv": ppv,
        "acc": accuracy,
        "predicted": float(len(predictions)),
        "gold": float(len(references)),
    }
