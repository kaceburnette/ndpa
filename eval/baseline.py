from typing import List, Tuple

from kernel.predictor import Predictor, WorkflowState


class LastNPredictor(Predictor):
    """
    Baseline: return the N most recently accessed files.

    This is the floor that the real kernel must beat.
    LRU is stronger than it looks on real coding sessions — pairs are usually
    cycling through a small file set. Beating it requires co-access graph features.
    """

    def __init__(self, n: int = 5):
        self.n = n

    def predict(self, state: WorkflowState, k: int = 5) -> List[Tuple[str, float]]:
        accesses = state.accesses if isinstance(state, WorkflowState) else state.get("accesses", [])
        if isinstance(state, dict):
            turn_idx = state.get("turn_idx", 0)
        else:
            turn_idx = state.turn_idx

        sorted_accesses = sorted(
            accesses,
            key=lambda a: a.get("last_turn", 0),
            reverse=True,
        )

        results = []
        for access in sorted_accesses[:k]:
            path = access["path"]
            recency = access.get("last_turn", 0)
            score = 1.0 / (1 + max(0, turn_idx - recency))
            results.append((path, score))
        return results
