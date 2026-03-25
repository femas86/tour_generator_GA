"""
ga/seeding.py — Inizializzazione della popolazione con greedy seeding.
Rispetta il TouristProfile in ogni costruzione:
  - Filtra le categorie non ammesse
  - Usa effective_score (con boost tag) nel criterio di selezione
  - Usa travel_time_min del profilo per i tempi
"""
from __future__ import annotations
import random
from config import ROUTE_DETOUR_FACTOR
from core.models import Individual, PoI
from core.distance import DistanceMatrix, haversine_km
from core.profile import TouristProfile
from ga.repair import RepairEngine


class GreedySeeder:

    def __init__(
        self,
        pois:       list[PoI],
        dm:         DistanceMatrix,
        repair:     RepairEngine,
        profile:    TouristProfile,
        start_time: int,
        budget:     int,
        start_lat:  float,
        start_lon:  float,
    ):
        self.pois       = pois
        self.dm         = dm
        self.repair     = repair
        self.profile    = profile
        self.start_time = start_time
        self.budget     = budget
        self.start_lat  = start_lat
        self.start_lon  = start_lon
        # Pool filtrato per categorie ammesse — usato in tutta la seeding
        self.allowed_pois = [
            p for p in pois
            if profile.allows_category(p.category.value)
        ]

    def build_population(self, pop_size: int) -> list[Individual]:
        population = []

        n_greedy    = max(1, int(pop_size * 0.20))
        n_perturbed = max(1, int(pop_size * 0.20))
        n_random    = pop_size - n_greedy - n_perturbed

        for _ in range(n_greedy):
            ind = self._greedy_construct(randomize=False, alpha=0.0)
            ind = self.repair.repair(ind)   # ← cap snack/ristoranti anche sui greedy
            population.append(ind)

        for i in range(n_perturbed):
            alpha = 0.15 + (i / n_perturbed) * 0.35
            ind   = self._greedy_construct(randomize=True, alpha=alpha)
            ind   = self.repair.repair(ind)   # ← idem
            population.append(ind)

        for _ in range(n_random):
            shuffled = random.sample(self.allowed_pois, len(self.allowed_pois))
            ind = Individual(genes=shuffled[:random.randint(1, len(shuffled))])
            ind = self.repair.repair(ind)
            population.append(ind)

        return population

    def _greedy_construct(self, randomize: bool = False, alpha: float = 0.0) -> Individual:
        """
        Greedy con RCL. Usa group_overhead per coerenza con FitnessEvaluator.
        Salta i ristoranti (li aggiunge _ensure_meal_slots) e riserva tempo
        solo per gli slot pasto SERALI (≥18:00), non per il pranzo — il pranzo
        cade nel flusso naturale del tour diurno e viene inserito da
        _ensure_meal_slots senza bisogno di riserva esplicita.
        """
        group_extra = max(0, self.profile.group_size - 1) * 5

        # Riserva tempo per ogni slot pasto che il greedy salta (tutti):
        # - slot serali (≥18:00): 90 min (cena dopo il tour diurno)
        # - slot diurni (<18:00): 75 min (pranzo inserito nel mezzo del tour)
        EVENING_RESERVE   = 90
        DAYTIME_RESERVE   = 75
        EVENING_THRESHOLD = 1080   # 18:00

        total_reserve = sum(
            EVENING_RESERVE if slot_open >= EVENING_THRESHOLD else DAYTIME_RESERVE
            for (slot_open, _) in self.profile.needs_meal_slot()
        )
        effective_end = self.start_time + self.budget - total_reserve

        tour      = []
        visited   = set()
        time_now  = self.start_time
        prev_lat  = self.start_lat
        prev_lon  = self.start_lon

        while True:
            candidates = []

            for poi in self.allowed_pois:
                if poi.id in visited:
                    continue
                if poi.category.value == "restaurant":
                    continue  # ristoranti: aggiunti da _ensure_meal_slots

                km         = haversine_km(prev_lat, prev_lon, poi.lat, poi.lon) * ROUTE_DETOUR_FACTOR
                travel_min = self.profile.travel_time_min(km)
                arrival    = time_now + travel_min

                if arrival > poi.time_window.close:
                    continue

                actual_arrival = max(arrival, poi.time_window.open)
                duration = poi.visit_duration + group_extra
                finish   = actual_arrival + duration
                if finish > effective_end:
                    continue

                overhead  = travel_min + max(0, poi.time_window.open - arrival)
                eff_score = self.profile.effective_score(poi)
                ratio     = eff_score / (overhead + duration + 1e-9)
                candidates.append((ratio, poi, actual_arrival, finish))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0], reverse=True)

            if randomize and len(candidates) > 1 and random.random() < alpha:
                rcl_size = max(1, int(len(candidates) * 0.20))
                _, poi, _, finish = random.choice(candidates[:rcl_size])
            else:
                _, poi, _, finish = candidates[0]

            tour.append(poi)
            visited.add(poi.id)
            prev_lat = poi.lat
            prev_lon = poi.lon
            time_now = finish

        return Individual(genes=tour)