import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Assignment = Dict[str, int]


# Process: Circular Distance Calculation
# Title: Compute shortest wrapped distance on a circular scale.
def circular_delta(value_a: float, value_b: float, period: float) -> float:
    raw = abs((value_a % period) - (value_b % period))
    return min(raw, period - raw)


# Process: Minute Hand Geometry
# Title: Convert minute value to angle in degrees.
def angle_for_minute(minute: int) -> float:
    return (minute % 60) * 6.0


# Process: Second Hand Geometry
# Title: Convert second value to angle in degrees.
def angle_for_second(second: int) -> float:
    return (second % 60) * 6.0


# Process: Hour Hand Geometry
# Title: Convert hour and minute to a continuous hour-hand angle.
def angle_for_hour(hour: int, minute: int) -> float:
    hour_12 = hour % 12
    return ((hour_12 + (minute % 60) / 60.0) * 30.0) % 360.0


# Process: Time Serialization
# Title: Format a canonical HH:MM:SS string.
def time_string(hour: int, minute: int, second: int) -> str:
    return f"{hour:02d}:{minute:02d}:{second:02d}"


@dataclass(frozen=True)
class WeightedClause:
    # Process: Weighted Rule Specification
    # Title: Represent a logical clause with score weight and hardness.
    name: str
    weight: float
    predicate: Callable[[Assignment], bool]
    hard: bool = False


class ProbabilisticLogicClockReasoner:
    """Formal probabilistic logic reasoner via weighted model enumeration.

    This is a compact weighted-model-counting style engine where each assignment
    (hour, minute, second) is a world, hard clauses remove invalid worlds,
    and soft weighted clauses score surviving worlds.
    """

    def __init__(
        self,
        observed_hour_angle: Optional[float],
        observed_minute_angle: Optional[float],
        observed_second_angle: Optional[float],
        candidate_angles_deg: Sequence[float],
    ):
        # Process: Observation and Candidate Initialization
        # Title: Persist observed hand angles and normalize candidate inputs.
        self.observed_hour_angle = observed_hour_angle
        self.observed_minute_angle = observed_minute_angle
        self.observed_second_angle = observed_second_angle
        self.candidate_angles_deg = [float(a) % 360.0 for a in candidate_angles_deg if isinstance(a, (float, int))]

    @staticmethod
    # Process: Candidate Angle Quantization
    # Title: Convert candidate angles into nearest minute ticks.
    def _minutes_from_angles(angles: Iterable[float]) -> List[int]:
        out = set()
        for angle in angles:
            out.add(int(round((float(angle) % 360.0) / 6.0)) % 60)
        return sorted(out)

    @staticmethod
    # Process: Hour Candidate Extraction
    # Title: Derive plausible neighboring hours from an hour-hand angle.
    def _hours_from_angle(angle: float) -> List[int]:
        raw_hour = ((angle % 360.0) / 30.0) % 12.0
        nearest = int(round(raw_hour)) % 12
        if nearest == 0:
            nearest = 12

        candidates = {nearest}
        candidates.add(12 if nearest == 1 else nearest - 1)
        candidates.add(1 if nearest == 12 else nearest + 1)
        return sorted(candidates)

    # Process: Domain Construction
    # Title: Build constrained value domains for hour, minute, and second.
    def _build_domains(self) -> Dict[str, List[int]]:
        minute_domain: set[int] = set()
        second_domain: set[int] = set()

        # Process: Minute Domain Expansion
        # Title: Expand around candidate minute values for tolerance.
        candidate_minutes = self._minutes_from_angles(self.candidate_angles_deg)
        for minute in candidate_minutes:
            minute_domain.update({minute, (minute - 1) % 60, (minute + 1) % 60})

        # Process: Minute Observation Anchoring
        # Title: Anchor minute domain around observed minute angle when available.
        if self.observed_minute_angle is not None:
            obs_m = int(round((self.observed_minute_angle % 360.0) / 6.0)) % 60
            minute_domain.update({obs_m, (obs_m - 1) % 60, (obs_m + 1) % 60, (obs_m + 2) % 60, (obs_m - 2) % 60})

        # Process: Second Domain Selection
        # Title: Build second candidates from observation or default to zero.
        if self.observed_second_angle is not None:
            obs_s = int(round((self.observed_second_angle % 360.0) / 6.0)) % 60
            second_domain.update({obs_s, (obs_s - 1) % 60, (obs_s + 1) % 60, (obs_s + 2) % 60, (obs_s - 2) % 60})
        else:
            second_domain.add(0)

        # Process: Hour Domain Selection
        # Title: Restrict hour candidates when an hour observation exists.
        hour_domain: set[int] = set(range(1, 13))
        if self.observed_hour_angle is not None:
            hour_domain = set(self._hours_from_angle(self.observed_hour_angle))

        # Process: Domain Fallback
        # Title: Provide a coarse minute grid when no minute candidates exist.
        if not minute_domain:
            minute_domain = set(range(0, 60, 5))

        return {
            "hour": sorted(hour_domain),
            "minute": sorted(minute_domain),
            "second": sorted(second_domain),
        }

    # Process: Rule Construction
    # Title: Define weighted logical constraints over candidate assignments.
    def _build_clauses(self, candidate_minutes: Sequence[int], candidate_hours: Sequence[int]) -> List[WeightedClause]:
        candidate_minutes_set = set(candidate_minutes)
        candidate_hours_set = set(candidate_hours)

        clauses: List[WeightedClause] = [
            WeightedClause(
                name="distinct_minute_second",
                weight=0.35,
                predicate=lambda a: a["minute"] != a["second"],
                hard=False,
            ),
            WeightedClause(
                name="prefer_candidate_minute",
                weight=0.55,
                predicate=lambda a: a["minute"] in candidate_minutes_set,
                hard=False,
            ),
            WeightedClause(
                name="prefer_candidate_hour",
                weight=0.45,
                predicate=lambda a: a["hour"] in candidate_hours_set,
                hard=False,
            ),
            WeightedClause(
                name="hour_minute_geometry_consistency",
                weight=0.9,
                predicate=lambda a: circular_delta(
                    angle_for_hour(a["hour"], a["minute"]),
                    self.observed_hour_angle if self.observed_hour_angle is not None else angle_for_hour(a["hour"], a["minute"]),
                    360.0,
                ) <= 25.0,
                hard=False,
            ),
        ]

        return clauses

    # Process: Measurement Likelihood
    # Title: Score an assignment using Gaussian-like angle residual penalties.
    def _measurement_log_score(self, assignment: Assignment) -> float:
        score = 0.0

        if self.observed_minute_angle is not None:
            minute_angle = angle_for_minute(assignment["minute"])
            delta = circular_delta(minute_angle, self.observed_minute_angle, 360.0)
            score += -0.5 * (delta / 9.5) ** 2

        if self.observed_second_angle is not None:
            second_angle = angle_for_second(assignment["second"])
            delta = circular_delta(second_angle, self.observed_second_angle, 360.0)
            score += -0.5 * (delta / 10.0) ** 2

        if self.observed_hour_angle is not None:
            expected_hour_angle = angle_for_hour(assignment["hour"], assignment["minute"])
            delta = circular_delta(expected_hour_angle, self.observed_hour_angle, 360.0)
            score += -0.5 * (delta / 16.0) ** 2

        return score

    # Process: Probabilistic Inference
    # Title: Enumerate worlds, weight them, and compute posterior outputs.
    def infer(self) -> Dict[str, object]:
        # Process: Setup
        # Title: Build domains, candidates, and weighted rules.
        domains = self._build_domains()
        candidate_minutes = self._minutes_from_angles(self.candidate_angles_deg)
        candidate_hours = self._hours_from_angle(self.observed_hour_angle) if self.observed_hour_angle is not None else list(range(1, 13))
        clauses = self._build_clauses(candidate_minutes, candidate_hours)

        worlds: List[Tuple[Assignment, float, List[Tuple[str, bool]]]] = []

        # Process: World Enumeration
        # Title: Evaluate every assignment and accumulate log-weights.
        for hour in domains["hour"]:
            for minute in domains["minute"]:
                for second in domains["second"]:
                    assignment: Assignment = {"hour": hour, "minute": minute, "second": second}

                    clause_outcomes: List[Tuple[str, bool]] = []
                    valid = True
                    log_weight = self._measurement_log_score(assignment)

                    for clause in clauses:
                        satisfied = bool(clause.predicate(assignment))
                        clause_outcomes.append((clause.name, satisfied))
                        if clause.hard and not satisfied:
                            valid = False
                            break
                        if satisfied:
                            log_weight += clause.weight

                    if valid:
                        worlds.append((assignment, log_weight, clause_outcomes))

        # Process: Empty-World Fallback
        # Title: Return a deterministic fallback when all worlds are invalid.
        if not worlds:
            fallback_assignment = {"hour": 12, "minute": 0, "second": 0}
            return {
                "method": "weighted-model-counting-enumeration",
                "world_count": 0,
                "domain_sizes": {k: len(v) for k, v in domains.items()},
                "map_assignment": {
                    **fallback_assignment,
                    "time": time_string(
                        fallback_assignment["hour"],
                        fallback_assignment["minute"],
                        fallback_assignment["second"],
                    ),
                },
                "map_probability": 0.0,
                "marginals": {"hour": [], "minute": [], "second": []},
                "rules": [],
            }

        # Process: Numerical Stabilization
        # Title: Convert log-weights to safe linear weights via max-shift.
        max_log_weight = max(w[1] for w in worlds)
        weighted_worlds: List[Tuple[Assignment, float, List[Tuple[str, bool]]]] = []
        total_weight = 0.0
        for assignment, log_weight, outcomes in worlds:
            weight = math.exp(log_weight - max_log_weight)
            weighted_worlds.append((assignment, weight, outcomes))
            total_weight += weight

        # Process: Posterior Normalization
        # Title: Normalize world weights into a probability distribution.
        posterior_worlds: List[Tuple[Assignment, float, List[Tuple[str, bool]]]] = []
        for assignment, weight, outcomes in weighted_worlds:
            posterior_worlds.append((assignment, weight / total_weight, outcomes))

        # Process: MAP Selection
        # Title: Pick the highest-probability assignment.
        map_assignment, map_prob, _ = max(posterior_worlds, key=lambda x: x[1])

        # Process: Marginal Aggregation
        # Title: Compute per-variable posterior marginals from world probabilities.
        marginals: Dict[str, Dict[int, float]] = {
            "hour": {},
            "minute": {},
            "second": {},
        }
        for assignment, prob, _ in posterior_worlds:
            for var in ("hour", "minute", "second"):
                value = assignment[var]
                marginals[var][value] = marginals[var].get(value, 0.0) + prob

        # Process: Marginal Formatting
        # Title: Keep only top-k values for concise output.
        def top_marginals(var: str, top_k: int = 5) -> List[Dict[str, float]]:
            items = sorted(marginals[var].items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            return [{"value": int(v), "probability": round(float(p), 6)} for v, p in items]

        # Process: Rule Diagnostics
        # Title: Summarize posterior mass satisfying each weighted clause.
        rule_summary: List[Dict[str, object]] = []
        for clause in clauses:
            satisfied_mass = 0.0
            for _, prob, outcomes in posterior_worlds:
                for name, satisfied in outcomes:
                    if name == clause.name and satisfied:
                        satisfied_mass += prob
                        break
            rule_summary.append(
                {
                    "name": clause.name,
                    "weight": clause.weight,
                    "hard": clause.hard,
                    "satisfied_posterior_mass": round(float(satisfied_mass), 6),
                }
            )

        # Process: Result Packaging
        # Title: Return MAP estimate, marginals, and rule diagnostics.
        return {
            "method": "weighted-model-counting-enumeration",
            "world_count": len(worlds),
            "domain_sizes": {k: len(v) for k, v in domains.items()},
            "map_assignment": {
                "hour": int(map_assignment["hour"]),
                "minute": int(map_assignment["minute"]),
                "second": int(map_assignment["second"]),
                "time": time_string(map_assignment["hour"], map_assignment["minute"], map_assignment["second"]),
            },
            "map_probability": round(float(map_prob), 6),
            "marginals": {
                "hour": top_marginals("hour", top_k=5),
                "minute": top_marginals("minute", top_k=5),
                "second": top_marginals("second", top_k=5),
            },
            "rules": rule_summary,
        }
