DEFAULT_SIGNAL_WEIGHTS = {
    "cabin_occupant_risk": 0.72,
    "phone_use": 0.64,
    "drowsy": 0.54,
    "driving_fatigue": 0.36,
    "eyes_closed": 0.34,
    "distracted": 0.34,
    "yawning": 0.22,
}

class RiskScorer:
    def __init__(self, weights: dict[str, float] = None):
        self.weights = weights or DEFAULT_SIGNAL_WEIGHTS

    def calculate_risk_score(self, active_signals: dict[str, float]) -> float:
        # 1. Hop nhat rui ro qua Noisy-OR
        prob_no_risk = 1.0
        for signal_name, signal_val in active_signals.items():
            if signal_val > 0 and signal_name in self.weights:
                weight = self.weights[signal_name]
                evidence = weight * signal_val
                prob_no_risk *= (1.0 - evidence)

        base_risk = 1.0 - prob_no_risk

        # 2. Boost theo ngu canh (Context Boost khi co to hop rui ro)
        context_boost = 0.0
        if active_signals.get("drowsy", 0) > 0 and active_signals.get("phone_use", 0) > 0:
            context_boost += 0.20
        if active_signals.get("distracted", 0) > 0 and active_signals.get("phone_use", 0) > 0:
            context_boost += 0.15

        final_risk = min(1.0, base_risk + context_boost)
        return round(final_risk, 3)