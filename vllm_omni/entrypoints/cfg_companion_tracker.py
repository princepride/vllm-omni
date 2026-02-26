"""Classifier-Free Guidance (CFG) companion request tracker.

When CFG is enabled for a multi-stage pipeline, a single user prompt may be
expanded into additional "companion" requests (e.g. a negative/empty prompt
for text-unconditional guidance).  These companions run through the initial
autoregressive stage in parallel with the primary request, and their KV
caches are later collected by downstream stages.

This module encapsulates all companion bookkeeping so that the main
orchestrator scheduling loop (``Omni._run_generation``) remains clean.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class CfgCompanionTracker:
    """Tracks CFG companion requests throughout their lifecycle.

    Responsibilities
    ----------------
    * **Prompt expansion** -- delegates to a model-specific
      ``prompt_expand_func`` to turn one user prompt into companion prompts.
    * **ID mapping** -- maintains bidirectional parent / companion mappings.
    * **Completion tracking** -- records which companions have finished Stage-0.
    * **Deferred parents** -- holds parent results while their companions
      are still running, and releases them once all companions complete.
    * **Failure propagation** -- marks parents as failed when a companion
      errors out, avoiding infinite waits.
    * **Timeout enforcement** -- expires parents that have waited too long
      for their companions.

    Usage (sketch inside ``Omni._run_generation``)::

        cfg = CfgCompanionTracker(expand_func, sp0)
        companions = cfg.expand_prompts(request_id_to_prompt)
        # ... submit companions to Stage-0 ...

        # In the polling loop:
        if cfg.is_companion(req_id):
            if error:
                parent_id, aborted = cfg.on_companion_error(req_id)
            else:
                ready_parent = cfg.on_companion_completed(req_id)
            continue

        if cfg.has_companions(req_id) and stage_id == 0:
            if cfg.all_companions_done(req_id):
                forward_parent(...)
            else:
                cfg.defer_parent(req_id, outputs, stage_id)
            continue

        timed_out = cfg.check_timeouts()
    """

    def __init__(
        self,
        prompt_expand_func: Callable[..., Any] | None,
        stage0_sampling_params: Any,
        timeout_s: float | None = None,
    ) -> None:
        self._expand_func = prompt_expand_func
        self._sp0 = stage0_sampling_params
        self._timeout_s = (
            timeout_s if timeout_s is not None else float(os.environ.get("VLLM_CFG_PENDING_TIMEOUT_S", "120"))
        )

        # parent_id -> {role: companion_id}
        self._companion_map: dict[str, dict[str, str]] = {}
        # All companion IDs (fast membership test)
        self._companion_ids: set[str] = set()
        # companion_id -> parent_id (reverse index)
        self._companion_to_parent: dict[str, str] = {}
        # parent_id -> set of completed companion IDs
        self._done: dict[str, set[str]] = {}
        # Deferred parent results awaiting companion completion
        self._pending_parents: dict[str, dict[str, Any]] = {}
        # Parents whose companions have failed
        self._failed_parents: set[str] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """``True`` after ``expand_prompts`` produced at least one companion."""
        return len(self._companion_ids) > 0

    @property
    def num_companions(self) -> int:
        """Total number of companion requests produced by expansion."""
        return len(self._companion_ids)

    @property
    def stage0_sampling_params(self) -> Any:
        """The Stage-0 sampling params used when submitting companions."""
        return self._sp0

    # ------------------------------------------------------------------
    # Prompt expansion
    # ------------------------------------------------------------------

    def expand_prompts(
        self,
        request_id_to_prompt: dict[str, Any],
    ) -> list[tuple[str, Any]]:
        """Expand user prompts into companion prompts.

        Delegates to the model-specific ``prompt_expand_func`` provided at
        construction time.

        Returns:
            List of ``(companion_id, companion_prompt)`` tuples ready for
            submission to Stage-0.  Empty when no expansion function is
            configured or no prompts require expansion.
        """
        if not self._expand_func:
            return []

        pairs: list[tuple[str, Any]] = []
        for rid, prompt in request_id_to_prompt.items():
            expanded = self._expand_func(prompt, self._sp0)
            if not expanded:
                continue
            role_map: dict[str, str] = {}
            for ep in expanded:
                cid = f"{rid}{ep.request_id_suffix}"
                role_map[ep.role] = cid
                self._companion_ids.add(cid)
                self._companion_to_parent[cid] = rid
                pairs.append((cid, ep.prompt))
            self._companion_map[rid] = role_map
            self._done[rid] = set()

        logger.debug(
            "CFG expansion: %d parent(s) -> %d companion(s)",
            len(self._companion_map),
            len(self._companion_ids),
        )
        return pairs

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_companion(self, req_id: str) -> bool:
        """Whether *req_id* is a companion (not user-facing)."""
        return req_id in self._companion_ids

    def has_companions(self, parent_id: str) -> bool:
        """Whether *parent_id* was expanded into companions."""
        return parent_id in self._companion_map

    def all_companions_done(self, parent_id: str) -> bool:
        """Whether every companion of *parent_id* has completed Stage-0."""
        role_map = self._companion_map.get(parent_id, {})
        done_set = self._done.get(parent_id, set())
        return all(cid in done_set for cid in role_map.values())

    def get_companion_request_ids(self, parent_id: str) -> dict[str, str]:
        """Return ``{role: companion_request_id}`` for *parent_id*."""
        return self._companion_map.get(parent_id, {})

    def is_parent_failed(self, parent_id: str) -> bool:
        """Whether a companion of *parent_id* has failed."""
        return parent_id in self._failed_parents

    # ------------------------------------------------------------------
    # Lifecycle events (called by the orchestrator loop)
    # ------------------------------------------------------------------

    def on_companion_error(self, companion_id: str) -> tuple[str | None, bool]:
        """Record a companion failure.

        Returns ``(parent_id, parent_was_aborted)`` where
        *parent_was_aborted* is ``True`` when the parent was already pending
        (deferred) and has now been removed -- the caller should count it as
        a completed (failed) request.
        """
        parent_id = self._companion_to_parent.get(companion_id)
        if parent_id is None:
            return None, False
        self._failed_parents.add(parent_id)
        logger.error(
            "CFG companion %s failed; marking parent %s as failed",
            companion_id,
            parent_id,
        )
        aborted = parent_id in self._pending_parents
        if aborted:
            self._pending_parents.pop(parent_id, None)
        return parent_id, aborted

    def on_companion_completed(self, companion_id: str) -> str | None:
        """Mark a companion as completed at Stage-0.

        Returns the *parent_id* when the parent is currently pending
        **and** all of its companions are now done (i.e. the parent is
        ready to be forwarded).  Otherwise returns ``None``.
        """
        parent_id = self._companion_to_parent.get(companion_id)
        if parent_id is None:
            return None
        self._done[parent_id].add(companion_id)
        logger.debug("CFG companion %s completed (parent=%s)", companion_id, parent_id)
        if parent_id in self._pending_parents and self.all_companions_done(parent_id):
            return parent_id
        return None

    def consume_parent_failure(self, parent_id: str) -> None:
        """Acknowledge a parent failure so it is not reported twice."""
        self._failed_parents.discard(parent_id)

    # ------------------------------------------------------------------
    # Deferred parent management
    # ------------------------------------------------------------------

    def defer_parent(
        self,
        parent_id: str,
        engine_outputs: Any,
        stage_id: int,
    ) -> None:
        """Store a parent's Stage-0 result while companions are still running."""
        self._pending_parents[parent_id] = {
            "engine_outputs": engine_outputs,
            "stage_id": stage_id,
            "pending_since": time.time(),
        }
        logger.debug("Parent %s deferred, waiting for CFG companions", parent_id)

    def pop_pending_parent(self, parent_id: str) -> dict[str, Any] | None:
        """Remove and return the deferred result for *parent_id*."""
        return self._pending_parents.pop(parent_id, None)

    def check_timeouts(self) -> list[str]:
        """Return parent IDs that exceeded the pending timeout.

        Timed-out parents are removed from the pending set.
        """
        if not self._pending_parents:
            return []
        now = time.time()
        timed_out: list[str] = []
        for pid in list(self._pending_parents):
            pending_since = self._pending_parents[pid].get("pending_since", now)
            if now - pending_since > self._timeout_s:
                self._pending_parents.pop(pid)
                self._failed_parents.discard(pid)
                timed_out.append(pid)
                logger.error(
                    "Parent %s timed out waiting for CFG companions (>%.0fs)",
                    pid,
                    self._timeout_s,
                )
        return timed_out
