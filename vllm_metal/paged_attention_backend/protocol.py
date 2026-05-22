# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PagedAttentionBackend(Protocol):
    def initialize(self, num_blocks: int) -> None: ...
    def patch_model(self, model: Any) -> int: ...
    def warm_up(self) -> None: ...
    def num_blocks(self) -> int: ...

    def mark_blocks_freed(self, block_ids: list[int]) -> None:
        """Queue freed block ranges on the per-layer ``ElasticKVPool``s.
        The physical page release happens on the next ``reclaim()``
        call. No-op when elastic mode is off."""
        ...

    def reclaim(self) -> int:
        """Release queued physical pages and refresh GPU mappings.
        Returns bytes released. No-op when elastic mode is off."""
        ...

    def get_stats(self) -> dict:
        """Return aggregated KV cache stats (sizing + elastic counters
        when applicable). Returns an empty dict if the backend's cache
        is not yet initialized."""
        ...
