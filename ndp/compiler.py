from .store import NDPStore


class ContextCompiler:
    """Packs staged NDP objects into a context string under a token budget."""

    def __init__(self, store: NDPStore, token_budget: int = 4000):
        self.store = store
        self.token_budget = token_budget

    def compile(self, tier: str = "hot") -> str:
        objects = self.store.get_by_tier(tier)
        parts = []
        used = 0
        for obj in objects:
            if obj.content is None:
                continue
            if used + obj.token_estimate > self.token_budget:
                continue
            parts.append(f"### {obj.source_path}\n{obj.content}")
            used += obj.token_estimate
        return "\n\n".join(parts)

    def compile_multi(self, tiers=("hot", "warm")) -> str:
        parts = []
        used = 0
        for tier in tiers:
            objects = self.store.get_by_tier(tier)
            for obj in objects:
                if obj.content is None:
                    continue
                if used + obj.token_estimate > self.token_budget:
                    return "\n\n".join(parts)
                parts.append(f"### {obj.source_path}\n{obj.content}")
                used += obj.token_estimate
        return "\n\n".join(parts)
