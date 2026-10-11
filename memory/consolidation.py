"""Consolidator: periodic sweep that promotes high-utility episodic items to the KG.

Steps:
1. Decay all items' utility.
2. Select promotable: utility >= threshold and not already consolidated.
3. Run the LLM-based fact extractor on the batch.
4. Upsert / reinforce facts in the KG.
5. Mark source items as consolidated.
6. Prune buffer back to capacity * prune_target_ratio (prefer consolidated items).
"""
from __future__ import annotations

import time

from memory.episodic import EpisodicBuffer
from memory.extractor import FactExtractor
from memory.kg import KGBackend
from memory.schemas import ConsolidationReport, MemoryItem
from memory.utility import UtilityScorer


class Consolidator:
    def __init__(
        self,
        buffer: EpisodicBuffer,
        kg: KGBackend,
        extractor: FactExtractor,
        promote_threshold: float = 0.6,
        decay_half_life_s: float = 3600.0,
        prune_target_ratio: float = 0.7,
        batch_cap: int = 20,
    ) -> None:
        self.buffer = buffer
        self.kg = kg
        self.extractor = extractor
        self.promote_threshold = promote_threshold
        self.decay_half_life_s = decay_half_life_s
        self.prune_target_ratio = prune_target_ratio
        self.batch_cap = batch_cap

    def run_once(self) -> ConsolidationReport:
        now = time.time()
        report = ConsolidationReport()

        items = self.buffer.all()
        # 1. Decay.
        for it in items:
            new = UtilityScorer.decay(it, now, self.decay_half_life_s)
            if abs(new - it.utility_score) > 1e-6:
                self.buffer.update_utility(it.id, new)
                report.decayed_count += 1

        # Refresh after decay.
        items = self.buffer.all()

        # 2. Select promotable.
        candidates = [
            it for it in items
            if it.utility_score >= self.promote_threshold and not it.consolidated
        ]
        candidates = sorted(candidates, key=lambda x: x.utility_score, reverse=True)[: self.batch_cap]
        report.candidates_considered = len(candidates)

        # 3+4. Extract + upsert.
        if candidates:
            try:
                facts = self.extractor.extract(candidates, batch_size=self.batch_cap)
                for f in facts:
                    self.kg.upsert_fact(f)
                report.promoted_facts = len(facts)
            except Exception:
                report.extraction_failed = True

            # 5. Mark consolidated regardless of fact count (the items were considered).
            self.buffer.mark_consolidated([it.id for it in candidates])

        # 6. Prune.
        target = max(1, int(self.buffer.capacity * self.prune_target_ratio))
        evicted = self.buffer.prune_to(target, prefer_consolidated=True)
        report.evicted_ids = evicted

        return report
