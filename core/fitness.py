"""
core/fitness.py — Valutazione fitness multi-obiettivo con profilo turista.
I tre obiettivi (score, distanza, tempo) vengono calcolati tenendo conto di:
  - tag_weights del profilo → boost/malus per score effettivo
  - transport_mode → velocità di spostamento corretta
  - want_lunch / want_dinner → penalità se manca il ristorante atteso
"""
from __future__ import annotations
from .models import Individual, FitnessScore, TourSchedule, ScheduledStop, PoICategory
from .distance import DistanceMatrix
from .profile import TouristProfile


class FitnessEvaluator:

    def __init__(
        self,
        dist_matrix:  DistanceMatrix,
        profile:      TouristProfile,
        start_time:   int,
        budget:       int,
        start_lat:    float,
        start_lon:    float,
        w_score:      float = 0.50,
        w_dist:       float = 0.20,
        w_time:       float = 0.30,
        penalty:      float = 50.0,
        meal_penalty: float = 0.25,
    ):
        self.dm           = dist_matrix
        self.profile      = profile
        self.start_time   = start_time
        self.budget       = budget
        self.start_lat    = start_lat
        self.start_lon    = start_lon
        self.w_score      = w_score
        self.w_dist       = w_dist
        self.w_time       = w_time
        self.penalty      = penalty
        self.meal_penalty = meal_penalty

    def decode(self, individual: Individual) -> TourSchedule:
        if individual._schedule is not None:
            return individual._schedule

        schedule      = TourSchedule()
        time_now      = self.start_time
        prev_lat      = self.start_lat
        prev_lon      = self.start_lon
        total_dist_km = 0.0
        total_wait_min = 0
        feasible      = True

        for poi in individual.genes:
            km         = self._km(prev_lat, prev_lon, poi)
            travel_min = self.profile.travel_time_min(km)
            total_dist_km += km

            arrival = time_now + travel_min

            if arrival > poi.time_window.close:
                feasible = False
                wait = 0
            else:
                wait    = max(0, poi.time_window.open - arrival)
                arrival = max(arrival, poi.time_window.open)

            # Visita più lunga in gruppo: +5 min per persona extra
            duration  = poi.visit_duration + max(0, self.profile.group_size - 1) * 5
            departure = arrival + duration
            time_now  = departure
            prev_lat  = poi.lat
            prev_lon  = poi.lon
            total_wait_min += wait   # accumula attese per penalità

            schedule.stops.append(ScheduledStop(
                poi=poi, arrival=arrival, departure=departure, wait=wait
            ))

        end_time = self.start_time + self.budget
        schedule.total_time     = time_now - self.start_time
        schedule.total_distance = round(total_dist_km, 2)
        schedule.total_wait     = total_wait_min
        schedule.is_feasible    = feasible and (time_now <= end_time)
        individual._schedule    = schedule
        return schedule

    def evaluate(self, individual: Individual) -> FitnessScore:
        schedule = self.decode(individual)
        end_time = self.start_time + self.budget

        # Score effettivo con boost da tag_weights del profilo
        total_score = sum(
            self.profile.effective_score(stop.poi)
            for stop in schedule.stops
        )

        time_over         = max(0, (self.start_time + schedule.total_time) - end_time)
        missing_meal_pen  = self._meal_coverage_penalty(schedule)

        # Normalizzazione basata sui PoI AMMESSI dal profilo (non sul totale)
        allowed_pois = [
            p for p in self.dm.pois
            if self.profile.allows_category(p.category.value)
        ]
        max_score = max(len(allowed_pois) * 1.5, 1.0)
        max_dist  = 50.0 if self.profile.transport_mode.value in ("car", "transit") else 15.0

        norm_score  = total_score / max_score
        norm_dist   = min(schedule.total_distance / max_dist, 1.0)
        time_over_h = (time_over / 60) * self.penalty
        # Penalizza attese eccessive (oltre 5 min totali): scoraggia tour con buchi
        total_wait  = getattr(schedule, 'total_wait', 0)
        wait_penalty = max(0, (total_wait - 5) / 60) * 10.0

        scalar = (
            self.w_score * norm_score
            - self.w_dist * norm_dist
            - time_over_h
            - wait_penalty
            - missing_meal_pen
        )

        fitness = FitnessScore(
            total_score    = round(total_score, 4),
            total_distance = schedule.total_distance,
            total_time     = schedule.total_time,
            is_feasible    = schedule.is_feasible,
            scalar         = round(scalar, 6),
        )
        individual.fitness = fitness
        return fitness

    def _meal_coverage_penalty(self, schedule: TourSchedule) -> float:
        """
        Penalità per ogni slot pasto richiesto dal profilo ma non coperto.
        NON si applica se il profilo non include la categoria ristorante:
        non ha senso penalizzare chi ha esplicitamente escluso i ristoranti.
        """
        if "restaurant" not in self.profile.allowed_categories:
            return 0.0

        penalty = 0.0
        for (slot_open, slot_close) in self.profile.needs_meal_slot():
            covered = any(
                stop.poi.category == PoICategory.RESTAURANT
                and slot_open <= stop.arrival <= slot_close
                for stop in schedule.stops
            )
            if not covered:
                penalty += self.meal_penalty
        return penalty

    def _km(self, lat: float, lon: float, poi) -> float:
        from .distance import haversine_km
        return haversine_km(lat, lon, poi.lat, poi.lon) * 1.3