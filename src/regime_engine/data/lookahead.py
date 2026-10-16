"""Empirical look-ahead detection.

Principle: if a transformation is causal, the values it produces for bars ``<= T`` cannot
depend on data after ``T``. ``LookAheadSanityChecker`` therefore:

1. computes features on the original data (the *reference*);
2. corrupts **every** row strictly after the cut-off ``T`` (several corruption modes, so a
   leak can't hide behind a corruption that happens to leave it unchanged);
3. recomputes the features and asserts that every value at or before ``T`` is **bit-for-bit
   identical** to the reference. NaN is treated as equal to NaN; any other difference,
   however small, is a failure.

The checker accepts any ``Callable[[DataFrame], DataFrame]``, so the same test covers the
feature engine, the rolling scaler, and any composition of them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum

import numpy as np
import pandas as pd

from regime_engine.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    LookAheadError,
)
from regime_engine.data.ingestor import OHLCV_COLUMNS
from regime_engine.utils.logging import LogEvent, get_logger

FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]

_LOG = get_logger("data.lookahead")


class CorruptionMode(str, Enum):
    RANDOM_WALK = "random_walk"  # replace the future with an unrelated, internally valid path
    SHUFFLE = "shuffle"  # permute future rows (same marginal distribution, scrambled order)
    SCALE = "scale"  # multiply future prices/volume by a large factor


@dataclass(frozen=True, slots=True)
class Divergence:
    """Where and how the post-corruption output differs from the reference at or before T."""

    first_divergent_ts: pd.Timestamp | None
    columns: tuple[str, ...]
    n_cells: int
    max_abs_diff: float
    structural: str | None = None
    mode: CorruptionMode | None = None
    trial: int | None = None

    def describe(self) -> str:
        where = f"[{self.mode.value} #{self.trial}] " if self.mode is not None else ""
        if self.structural is not None:
            return f"{where}{self.structural}"
        return (
            f"{where}{self.n_cells} cell(s) differ in {list(self.columns)}; first at "
            f"{self.first_divergent_ts}; max |diff| = {self.max_abs_diff:.3e}"
        )


@dataclass(frozen=True, slots=True)
class LookAheadReport:
    cutoff: pd.Timestamp
    n_history_rows: int
    n_future_rows: int
    trials: tuple[CorruptionMode, ...]
    divergences: tuple[Divergence, ...]

    @property
    def passed(self) -> bool:
        return not self.divergences

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        head = (
            f"look-ahead check {status} at cutoff {self.cutoff}: {self.n_history_rows} "
            f"history rows compared, {self.n_future_rows} future rows corrupted, "
            f"{len(self.trials)} trial(s)"
        )
        return "\n  ".join([head, *(d.describe() for d in self.divergences)])


class LookAheadSanityChecker:
    """Corrupt everything after T; require outputs at or before T to be bit-identical."""

    def __init__(
        self,
        feature_fn: FeatureFn,
        *,
        modes: Sequence[CorruptionMode] = tuple(CorruptionMode),
        trials_per_mode: int = 1,
        seed: int = 0,
    ) -> None:
        if not modes:
            raise ValueError("at least one corruption mode is required")
        if trials_per_mode < 1:
            raise ValueError("trials_per_mode must be >= 1")
        self._feature_fn = feature_fn
        self._modes = tuple(modes)
        self._trials = trials_per_mode
        self._seed = seed

    # ---- corruption ----------------------------------------------------------------------

    @staticmethod
    def normalize_cutoff(cutoff: pd.Timestamp | date | str, index: pd.Index) -> pd.Timestamp:
        ts = pd.Timestamp(cutoff)
        tz = getattr(index, "tz", None)
        if tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        elif tz is None and ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts

    @staticmethod
    def corrupt(
        data: pd.DataFrame,
        cutoff: pd.Timestamp,
        mode: CorruptionMode,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Return a copy of ``data`` with every OHLCV row strictly after ``cutoff`` replaced.

        Corrupted rows stay *internally valid* (positive prices, high >= max(open, close),
        low <= min(open, close), positive volume), so a correct feature function never fails
        on them. Any change at or before ``cutoff`` can only come from a leak.
        """
        missing = [c for c in OHLCV_COLUMNS if c not in data.columns]
        if missing:
            raise DataValidationError(f"cannot corrupt: missing columns {missing}")
        future = np.asarray(data.index > cutoff, dtype=bool)
        n = int(future.sum())
        if n == 0:
            raise InsufficientDataError(f"no rows after cutoff {cutoff}; nothing to corrupt")

        cols = list(OHLCV_COLUMNS)
        out = data.copy()
        original = data.loc[future, cols].to_numpy(dtype=np.float64)

        if mode is CorruptionMode.RANDOM_WALK:
            history_close = data.loc[~future, "close"]
            anchor = float(history_close.iloc[-1]) if len(history_close) else float(original[0, 3])
            close = anchor * np.exp(np.cumsum(rng.normal(0.0, 0.05, n)))
            open_ = close * np.exp(rng.normal(0.0, 0.02, n))
            high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.02, n)))
            low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.02, n)))
            med_volume = float(np.median(original[:, 4])) if n else 1.0
            volume = np.exp(rng.normal(np.log(max(med_volume, 1.0)), 1.0, n))
            replacement = np.column_stack([open_, high, low, close, volume])
        elif mode is CorruptionMode.SHUFFLE:
            perm = rng.permutation(n)
            if n > 1 and np.array_equal(perm, np.arange(n)):
                perm = np.roll(perm, 1)
            replacement = original[perm]
        elif mode is CorruptionMode.SCALE:
            factor = 10.0 ** rng.uniform(1.0, 3.0)
            replacement = original.copy()
            replacement[:, :4] *= factor
            replacement[:, 4] *= factor * 10.0
        else:  # pragma: no cover - exhaustive over the enum
            raise ValueError(f"unknown corruption mode {mode!r}")

        out.loc[future, cols] = replacement

        # Self-check: the corruption must never touch history.
        if not out.loc[~future].equals(data.loc[~future]):
            raise RuntimeError("internal error: corruption modified rows at or before cutoff")
        return out

    # ---- comparison ----------------------------------------------------------------------

    @staticmethod
    def compare(
        reference: pd.DataFrame, candidate: pd.DataFrame, cutoff: pd.Timestamp
    ) -> Divergence | None:
        """Exact comparison of the rows at or before ``cutoff``. ``None`` means identical."""
        ref = reference.loc[np.asarray(reference.index <= cutoff, dtype=bool)]
        cand = candidate.loc[np.asarray(candidate.index <= cutoff, dtype=bool)]

        if list(ref.columns) != list(cand.columns):
            return Divergence(
                None,
                (),
                0,
                float("inf"),
                structural=f"columns differ: {list(ref.columns)} vs {list(cand.columns)}",
            )
        if not ref.index.equals(cand.index):
            changed = ref.index.symmetric_difference(cand.index)
            return Divergence(
                changed.min() if len(changed) else None,
                (),
                0,
                float("inf"),
                structural=(
                    f"history index differs ({len(ref)} vs {len(cand)} rows, "
                    f"{len(changed)} timestamp(s) not shared)"
                ),
            )

        a = ref.to_numpy(dtype=np.float64)
        b = cand.to_numpy(dtype=np.float64)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if same.all():
            return None

        differs = ~same
        rows = np.flatnonzero(differs.any(axis=1))
        with np.errstate(invalid="ignore"):
            abs_diff = np.where(differs, np.abs(a - b), np.nan)
        # A NaN-vs-number mismatch has an undefined magnitude; report it as infinite.
        abs_diff = np.where(differs & np.isnan(abs_diff), np.inf, abs_diff)
        return Divergence(
            first_divergent_ts=ref.index[rows[0]],
            columns=tuple(str(c) for c in ref.columns[differs.any(axis=0)]),
            n_cells=int(differs.sum()),
            max_abs_diff=float(np.nanmax(abs_diff)),
        )

    # ---- driver --------------------------------------------------------------------------

    def run(self, data: pd.DataFrame, cutoff: pd.Timestamp | date | str) -> LookAheadReport:
        ts = self.normalize_cutoff(cutoff, data.index)
        pristine = data.copy()
        reference = self._feature_fn(data.copy())
        n_history = int(np.asarray(reference.index <= ts, dtype=bool).sum())
        if n_history == 0:
            raise InsufficientDataError(
                f"feature output has no rows at or before cutoff {ts} "
                "(cutoff falls inside the warm-up period)"
            )
        n_future = int(np.asarray(data.index > ts, dtype=bool).sum())

        rng = np.random.default_rng(self._seed)
        trials: list[CorruptionMode] = []
        divergences: list[Divergence] = []
        for mode in self._modes:
            for trial in range(self._trials):
                corrupted = self.corrupt(data, ts, mode, rng)
                snapshot = corrupted.copy()
                candidate = self._feature_fn(corrupted)
                trials.append(mode)
                if not corrupted.equals(snapshot):
                    divergences.append(
                        Divergence(
                            None,
                            (),
                            0,
                            float("inf"),
                            structural="feature function mutated its input frame",
                            mode=mode,
                            trial=trial,
                        )
                    )
                divergence = self.compare(reference, candidate, ts)
                if divergence is not None:
                    divergences.append(dataclasses.replace(divergence, mode=mode, trial=trial))

        if not data.equals(pristine):
            raise RuntimeError("internal error: input data was modified during the check")

        report = LookAheadReport(ts, n_history, n_future, tuple(trials), tuple(divergences))
        if report.passed:
            _LOG.info(
                LogEvent.LOOKAHEAD_CHECK_PASSED,
                "no look-ahead detected",
                cutoff=ts,
                history_rows=n_history,
                future_rows=n_future,
                trials=len(trials),
            )
        else:
            _LOG.error(
                LogEvent.LOOKAHEAD_CHECK_FAILED,
                "look-ahead detected",
                cutoff=ts,
                divergences=[d.describe() for d in divergences],
            )
        return report

    def assert_no_lookahead(
        self, data: pd.DataFrame, cutoff: pd.Timestamp | date | str
    ) -> LookAheadReport:
        report = self.run(data, cutoff)
        if not report.passed:
            raise LookAheadError(report.summary())
        return report
