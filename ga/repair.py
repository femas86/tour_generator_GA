"""
ga/repair.py — Motore di riparazione genetica.
Ripara individui infeasible rimuovendo o riposizionando i PoI che
violano le time window o sforano il budget giornaliero.
"""
from __future__ import annotations
from core.models import Individual, PoI
from core.distance import DistanceMatrix


class RepairEngine:
    """
    Strategie di riparazione (applicate in sequenza):
      1. repair_time_windows:   riordina per earliest deadline first,
                                poi rimuove i PoI irrecuperabili.
      2. repair_budget:         rimuove iterativamente il PoI con il
                                peggior rapporto score/visit_duration.
    """

    def __init__(
        self,
        dm:          DistanceMatrix,
        start_time:  int,
        budget:      int,
        start_lat:   float,
        start_lon:   float,
    ):
        self.dm         = dm
        self.start_time = start_time
        self.budget     = budget
        self.start_lat  = start_lat
        self.start_lon  = start_lon

    def repair(self, individual: Individual) -> Individual:
        """Pipeline completa: riordina → fix TW → fix budget."""
        individual.invalidate_cache()
        individual = self._sort_by_earliest_deadline(individual)
        individual = self.repair_time_windows(individual)
        individual = self.repair_budget(individual)
        individual.invalidate_cache()
        return individual

    def _sort_by_earliest_deadline(self, individual: Individual) -> Individual:
        """
        Riordina i PoI per Earliest Deadline First (EDF):
        i PoI che chiudono prima vanno visitati prima.
        Questo riduce drasticamente le violazioni di TW.
        """
        individual.genes.sort(key=lambda p: p.time_window.close)
        return individual

    def repair_time_windows(self, individual: Individual) -> Individual:
        """
        Simula il tour e rimuove i PoI la cui time window è già scaduta
        al momento dell'arrivo previsto.
        """
        valid  = []
        time_now  = self.start_time
        prev_lat  = self.start_lat
        prev_lon  = self.start_lon

        for poi in individual.genes:
            travel = self._travel_time(prev_lat, prev_lon, poi)
            arrival = time_now + travel

            if arrival > poi.time_window.close:
                continue  # impossibile: salta il PoI

            # Attesa se si arriva troppo presto
            arrival   = max(arrival, poi.time_window.open)
            departure = arrival + poi.visit_duration

            valid.append(poi)
            time_now = departure
            prev_lat = poi.lat
            prev_lon = poi.lon

        individual.genes = valid
        return individual

    def repair_budget(self, individual: Individual) -> Individual:
        """
        Rimuove iterativamente il PoI con il peggior rapporto
        score/visit_duration finché il tour rientra nel budget.
        """
        while True:
            total = self._simulate_total_time(individual.genes)
            if total <= self.budget or not individual.genes:
                break
            # Candidato da rimuovere: minor valore per minuto occupato
            worst = min(
                individual.genes,
                key=lambda p: p.score / (p.visit_duration + 1e-9)
            )
            individual.genes.remove(worst)

        return individual

    def _simulate_total_time(self, genes: list[PoI]) -> int:
        """Calcola il tempo totale del tour senza costruire lo schedule completo."""
        time_now = self.start_time
        prev_lat = self.start_lat
        prev_lon = self.start_lon

        for poi in genes:
            travel  = self._travel_time(prev_lat, prev_lon, poi)
            arrival = time_now + travel
            arrival = max(arrival, poi.time_window.open)
            time_now = arrival + poi.visit_duration
            prev_lat = poi.lat
            prev_lon = poi.lon

        return time_now - self.start_time

    def _travel_time(self, lat: float, lon: float, to: PoI) -> int:
        """Wrapper: usa la matrice se disponibile, altrimenti Haversine."""
        from core.distance import haversine_km
        speed_kmh = 4.5  # a piedi
        km = haversine_km(lat, lon, to.lat, to.lon) * 1.3
        return int((km / speed_kmh) * 60)