"""
ga/seeding.py — Inizializzazione della popolazione con greedy seeding.
Combina costruzione greedy deterministica, α-greedy perturbata
e individui casuali riparati per massimizzare diversità e qualità iniziale.
"""
from __future__ import annotations
import random
from core.models import Individual, PoI
from core.distance import DistanceMatrix, haversine_km
from ga.repair import RepairEngine


class GreedySeeder:
    """
    Costruisce la popolazione iniziale con strategia mista:
      - 20% greedy puro (deterministico, massima qualità)
      - 20% α-greedy perturbato (qualità buona + diversità)
      - 60% casuale riparato (massima diversità genetica)
    """

    def __init__(
        self,
        pois:       list[PoI],
        dm:         DistanceMatrix,
        repair:     RepairEngine,
        start_time: int,
        budget:     int,
        start_lat:  float,
        start_lon:  float,
    ):
        self.pois       = pois
        self.dm         = dm
        self.repair     = repair
        self.start_time = start_time
        self.budget     = budget
        self.start_lat  = start_lat
        self.start_lon  = start_lon

    def build_population(self, pop_size: int) -> list[Individual]:
        population = []

        n_greedy    = max(1, int(pop_size * 0.20))
        n_perturbed = max(1, int(pop_size * 0.20))
        n_random    = pop_size - n_greedy - n_perturbed

        # 1. Greedy puro
        for _ in range(n_greedy):
            ind = self._greedy_construct(randomize=False, alpha=0.0)
            population.append(ind)

        # 2. α-greedy perturbato (diversi valori di alpha)
        for i in range(n_perturbed):
            alpha = 0.15 + (i / n_perturbed) * 0.35  # da 0.15 a 0.50
            ind   = self._greedy_construct(randomize=True, alpha=alpha)
            population.append(ind)

        # 3. Casuali riparati
        for _ in range(n_random):
            shuffled = random.sample(self.pois, len(self.pois))
            ind = Individual(genes=shuffled[:random.randint(1, len(shuffled))])
            ind = self.repair.repair(ind)
            population.append(ind)

        return population

    def _greedy_construct(self, randomize: bool = False, alpha: float = 0.0) -> Individual:
        """
        Costruzione greedy con Restricted Candidate List (RCL).
        
        Criterio di selezione: ratio = score / (overhead_time + visit_duration)
        dove overhead_time include spostamento + eventuale attesa all'apertura.
        
        alpha: probabilità di scegliere casualmente dalla RCL invece del best.
               alpha=0  → greedy deterministico
               alpha=0.5 → semi-casuale (GRASP-like)
        """
        tour      = []
        visited   = set()
        time_now  = self.start_time
        prev_lat  = self.start_lat
        prev_lon  = self.start_lon
        end_time  = self.start_time + self.budget

        while True:
            candidates = []

            for poi in self.pois:
                if poi.id in visited:
                    continue

                travel_min = self._travel_time(prev_lat, prev_lon, poi)
                arrival    = time_now + travel_min

                # Salta se la TW è già scaduta
                if arrival > poi.time_window.close:
                    continue

                # Salta se non si riesce a completare la visita entro il budget
                actual_arrival = max(arrival, poi.time_window.open)
                finish = actual_arrival + poi.visit_duration
                if finish > end_time:
                    continue

                # Calcola overhead (spostamento + attesa)
                overhead = travel_min + max(0, poi.time_window.open - arrival)
                ratio    = poi.score / (overhead + poi.visit_duration + 1e-9)
                candidates.append((ratio, poi, actual_arrival, finish))

            if not candidates:
                break  # tour saturo

            # Ordina per ratio decrescente
            candidates.sort(key=lambda x: x[0], reverse=True)

            if randomize and len(candidates) > 1 and random.random() < alpha:
                # Scegli casualmente dalla RCL (top 20% per ratio)
                rcl_size   = max(1, int(len(candidates) * 0.20))
                _, poi, _, finish = random.choice(candidates[:rcl_size])
            else:
                _, poi, _, finish = candidates[0]

            tour.append(poi)
            visited.add(poi.id)
            prev_lat = poi.lat
            prev_lon = poi.lon
            time_now = finish

        return Individual(genes=tour)

    def _travel_time(self, lat: float, lon: float, to: PoI) -> int:
        speed_kmh = 4.5
        km = haversine_km(lat, lon, to.lat, to.lon) * 1.3
        return int((km / speed_kmh) * 60)