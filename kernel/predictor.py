"""
Heuristic prediction kernel.

Scores files by likelihood of being needed in the next N turns.
No LLM calls. No blocking I/O. Target: <100ms per predict() call.

Scoring features:
  recency           — how recently accessed this session
  frequency         — how many times accessed this session
  extension_affinity — test/spec/type pairing rules
  import_proximity  — files imported by currently hot files
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

WEIGHTS = {
    "recency": 0.35,
    "frequency": 0.25,
    "extension_affinity": 0.20,
    "import_proximity": 0.20,
}

AFFINITY_MAP = {
    ".ts":   [".test.ts", ".spec.ts", ".d.ts"],
    ".tsx":  [".test.tsx", ".spec.tsx", ".stories.tsx"],
    ".js":   [".test.js", ".spec.js"],
    ".jsx":  [".test.jsx", ".spec.jsx"],
    ".py":   ["_test.py"],
    ".go":   ["_test.go"],
    ".rs":   ["_test.rs"],
    ".c":    [".h"],
    ".cpp":  [".h", ".hpp"],
    ".css":  [".module.css", ".scss"],
}

IMPORT_PATTERNS: Dict[str, List[str]] = {
    ".ts":   [r'from\s+[\'"](\.[^\'\"]+)[\'"]', r'require\s*\(\s*[\'"](\.[^\'\"]+)[\'"]'],
    ".tsx":  [r'from\s+[\'"](\.[^\'\"]+)[\'"]'],
    ".js":   [r'from\s+[\'"](\.[^\'\"]+)[\'"]', r'require\s*\(\s*[\'"](\.[^\'\"]+)[\'"]'],
    ".jsx":  [r'from\s+[\'"](\.[^\'\"]+)[\'"]'],
    ".py":   [r'^from\s+(\.\S+)\s+import', r'^import\s+(\.\S+)'],
    ".go":   [],
    ".rs":   [],
}


@dataclass
class WorkflowState:
    accesses: list = field(default_factory=list)
    cwd: str = ""
    turn_idx: int = 0


class Predictor(ABC):
    @abstractmethod
    def predict(self, state, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k (path, score) pairs. Higher = more likely needed."""
        ...


class NullPredictor(Predictor):
    def predict(self, state, k: int = 5) -> List[Tuple[str, float]]:
        return []


class HeuristicPredictor(Predictor):
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or WEIGHTS
        self._import_cache: Dict[str, Set[str]] = {}

    def predict(self, state, k: int = 5) -> List[Tuple[str, float]]:
        if isinstance(state, dict):
            accesses = state.get("accesses", [])
            turn_idx = state.get("turn_idx", 0)
            cwd = state.get("cwd", "")
        else:
            accesses = state.accesses
            turn_idx = state.turn_idx
            cwd = state.cwd

        if not accesses:
            return []

        # "hot" = 3 most recently touched files — drives affinity + import scoring
        hot_files: Set[str] = {
            a["path"]
            for a in sorted(accesses, key=lambda a: a.get("last_turn", 0), reverse=True)[:3]
        }

        # Candidate set: everything accessed + affinity + import neighbors
        candidates: Set[str] = {a["path"] for a in accesses}
        for path in list(hot_files):
            candidates.update(self._affinity_candidates(path))
            candidates.update(self._get_imports(path, cwd))

        access_map = {a["path"]: a for a in accesses}

        # Drop candidates that don't exist and haven't been accessed before
        seen_paths = {a["path"] for a in accesses}
        candidates = {p for p in candidates if p in seen_paths or Path(p).exists()}

        scores: Dict[str, float] = {}
        for path in candidates:
            acc = access_map.get(path)
            scores[path] = (
                self.weights["recency"]           * self._recency(acc, turn_idx)
                + self.weights["frequency"]       * self._frequency(acc)
                + self.weights["extension_affinity"] * self._extension(path, hot_files)
                + self.weights["import_proximity"]   * self._import_prox(path, hot_files, cwd)
            )

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    # --- feature scorers ---

    def _recency(self, acc, turn_idx: int) -> float:
        if acc is None:
            return 0.0
        gap = max(0, turn_idx - acc.get("last_turn", 0))
        return 1.0 / (1 + gap)

    def _frequency(self, acc) -> float:
        if acc is None:
            return 0.0
        return min(1.0, acc.get("count", 0) / 5.0)

    def _extension(self, path: str, hot_files: Set[str]) -> float:
        p = Path(path)
        score = 0.0
        for hot in hot_files:
            hp = Path(hot)
            if hp.stem == p.stem and hp.suffix != p.suffix:
                score = max(score, 0.85)
            for affinite in AFFINITY_MAP.get(hp.suffix, []):
                if path.endswith(affinite) and hp.stem in p.stem:
                    score = max(score, 0.65)
        return score

    def _import_prox(self, path: str, hot_files: Set[str], cwd: str) -> float:
        for hot in hot_files:
            imports = self._get_imports(hot, cwd)
            for imp in imports:
                if imp == path or path.endswith(imp):
                    return 0.9
        return 0.0

    # --- import parsing ---

    def _get_imports(self, file_path: str, cwd: str) -> Set[str]:
        if file_path in self._import_cache:
            return self._import_cache[file_path]

        result: Set[str] = set()
        p = Path(file_path)
        patterns = IMPORT_PATTERNS.get(p.suffix, [])

        if patterns and p.exists() and p.is_file():
            try:
                content = p.read_text(errors="replace")
                for pattern in patterns:
                    for m in re.finditer(pattern, content, re.MULTILINE):
                        raw = m.group(1)
                        if raw.startswith("."):
                            base = (p.parent / raw).resolve()
                            result.add(str(base))
                            for ext in (".ts", ".tsx", ".js", ".jsx", ".py"):
                                result.add(str(base) + ext)
                                result.add(str(base / "index") + ext)
            except Exception:
                pass

        self._import_cache[file_path] = result
        return result

    def _affinity_candidates(self, path: str) -> List[str]:
        p = Path(path)
        out = []
        for ext in AFFINITY_MAP.get(p.suffix, []):
            out.append(str(p.parent / (p.stem + ext)))
        return out
