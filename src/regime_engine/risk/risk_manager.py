"""The automated risk desk.

Per-bar call order (the backtester follows it):

    start_bar()            advance cooldown countdowns
    check_stops(bar)       intrabar stop/target check, gap-aware (arms the stop lockout)
    on_regime(state)       arms the regime-flip cooldown on a confirmed change
    evaluate(target, st)   ordered rules -> RiskDecision with a full audit trace
    on_position(...)       after fills: (re)set protective levels
    on_close(bar, atr)     ratchet trailing stops

Rule order: drawdown breaker -> regime cooldown -> stop lockout -> VaR budget -> leverage.
The breaker comes first so it can always flatten. Cooldowns only block risk *increases*, so
protective exits are never delayed.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from regime_engine.core.config import EngineConfig
from regime_engine.core.types import (
    RegimeState,
    RiskAction,
    RiskDecision,
    RuleOutcome,
    TargetPosition,
)
from regime_engine.risk.rules import (
    CooldownRule,
    DrawdownBreaker,
    LeverageRule,
    RiskRule,
    RiskState,
    VaRBudgetRule,
)
from regime_engine.risk.stops import StopManager, StopTrigger
from regime_engine.risk.var import VaRCalculator
from regime_engine.utils.logging import LogEvent, get_logger

_LOG = get_logger("risk.manager")


class RiskManager:
    def __init__(
        self,
        rules: Sequence[RiskRule],
        *,
        cooldown: CooldownRule,
        stop_lockout: CooldownRule,
        stops: StopManager,
        var: VaRCalculator,
    ) -> None:
        if cooldown not in rules or stop_lockout not in rules:
            raise ValueError("cooldown and stop_lockout must be part of the rule list")
        self._rules = tuple(rules)
        self._cooldown = cooldown
        self._lockout = stop_lockout
        self._stops = stops
        self._var = var

    @classmethod
    def from_config(cls, config: EngineConfig) -> RiskManager:
        risk = config.risk
        cooldown = CooldownRule(config.cooldown.bars, config.cooldown.mode, name="regime_cooldown")
        lockout = CooldownRule(risk.stop_lockout_bars, name="stop_lockout")
        rules: list[RiskRule] = [
            DrawdownBreaker(risk.dd_breaker, risk.dd_halt_bars),
            cooldown,
            lockout,
            VaRBudgetRule(risk.var_budget),
            LeverageRule(config.sizing.max_leverage, allow_short=risk.allow_short),
        ]
        return cls(
            rules,
            cooldown=cooldown,
            stop_lockout=lockout,
            stops=StopManager(risk.trailing_stop_atr, risk.profit_target_atr),
            var=VaRCalculator(risk.var_alpha, risk.var_window),
        )

    @property
    def rules(self) -> tuple[RiskRule, ...]:
        return self._rules

    @property
    def var(self) -> VaRCalculator:
        return self._var

    @property
    def stops(self) -> StopManager:
        return self._stops

    @property
    def cooldown_active(self) -> bool:
        return self._cooldown.active

    # ---- per-bar lifecycle -------------------------------------------------------------------

    def start_bar(self, ts: pd.Timestamp) -> None:
        for rule in (self._cooldown, self._lockout):
            if rule.tick():
                _LOG.info(LogEvent.COOLDOWN_END, f"{rule.name} ended", ts=ts, rule=rule.name)

    def check_stops(
        self, ts: pd.Timestamp, open_: float, high: float, low: float
    ) -> StopTrigger | None:
        trigger = self._stops.check_bar(ts, open_, high, low)
        if trigger is not None:
            _LOG.warning(
                LogEvent.STOP_TRIGGERED,
                f"{trigger.kind.value} hit",
                ts=ts,
                side=trigger.side.name,
                level=trigger.level,
                fill_price=trigger.fill_price,
                gapped=trigger.fill_price != trigger.level,
            )
            self._lockout.arm()
            if self._lockout.active:
                _LOG.info(
                    LogEvent.COOLDOWN_START,
                    "stop lockout started",
                    ts=ts,
                    rule=self._lockout.name,
                    bars=self._lockout.remaining,
                )
        return trigger

    def on_regime(self, state: RegimeState | None) -> None:
        if state is not None and state.changed:
            self._cooldown.arm()
            if self._cooldown.active:
                _LOG.info(
                    LogEvent.COOLDOWN_START,
                    f"regime flip to {state.regime.name}: cooldown started",
                    ts=state.ts,
                    rule=self._cooldown.name,
                    bars=self._cooldown.remaining,
                )

    def evaluate(self, requested: TargetPosition, state: RiskState) -> RiskDecision:
        weight = requested.weight
        trace: list[RuleOutcome] = []
        for rule in self._rules:
            allowed, reason = rule.apply(weight, state)
            if allowed == weight:
                action = RiskAction.PASS
            elif allowed == 0.0:
                action = RiskAction.VETO
            else:
                action = RiskAction.MODIFY
            trace.append(RuleOutcome(rule.name, action, weight, allowed, reason))
            if action is not RiskAction.PASS:
                _LOG.debug(
                    LogEvent.RISK_VETO if action is RiskAction.VETO else LogEvent.RISK_MODIFY,
                    reason,
                    ts=state.ts,
                    rule=rule.name,
                    before=weight,
                    after=allowed,
                )
            weight = allowed
        approved = TargetPosition(
            requested.ts,
            weight,
            requested.rationale if weight == requested.weight else "risk-adjusted",
            requested.components,
        )
        return RiskDecision(requested.ts, requested, approved, tuple(trace))

    def on_position(self, weight: float, price: float, atr: float) -> None:
        self._stops.on_position(weight, price, atr)

    def on_close(self, high: float, low: float, atr: float) -> None:
        self._stops.on_close(high, low, atr)
