"""
core/fitness.py — Valutazione fitness multi-obiettivo.
Decodifica il cromosoma in uno schedule concreto con orari,
poi calcola i tre obiettivi (score, distanza, tempo).
"""
from __future__ import annotations
from .models import Individual, FitnessScore, TourSchedule, ScheduledStop
from .distance import DistanceMatrix


class FitnessEvaluator:
    """
    Valuta un Individual calcolando:
      - total_score:    somma score PoI visitati (da massimizzare)
      - total_distance: km totali percorsi (da minimizzare)
      - total_time:     minuti totali usati (da minimizzare / penalizzare)
    
    La funzione scalare aggregata è usata come fallback per selezioni
    rapide (tournament selection) e nei test. Per NSGA-II si usano
    direttamente i tre obiettivi separati.
    """

    def __init__(
        self,
        dist_matrix:  DistanceMatrix,
        start_time:   int,           # minuti: es. 540 = 09:00
        budget:       int,           # minuti: es. 480 = 8 ore
        start_lat:    float,
        start_lon:    float,
        w_score:      float = 0.5,
        w_dist:       float = 0.2,
        w_time:       float = 0.3,
        penalty:      float = 50.0,  # penalità per minuto di sforamento
    ):
        self.dm         = dist_matrix
        self.start_time = start_time
        self.budget     = budget
        self.start_lat  = start_lat
        self.start_lon  = start_lon
        self.w_score    = w_score
        self.w_dist     = w_dist
        self.w_time     = w_time
        self.penalty    = penalty

    def decode(self, individual: Individual) -> TourSchedule:
        """
        Converte la lista di PoI in uno schedule con orari concreti.
        Gestisce attese alle time window e accumula distanza/tempo.
        """
        if individual._schedule is not None:
            return individual._schedule

        schedule      = TourSchedule()
        time_now      = self.start_time
        prev_lat      = self.start_lat
        prev_lon      = self.start_lon
        total_dist_km = 0.0
        feasible      = True

        for poi in individual.genes:
            # Tempo di spostamento dal nodo precedente
            travel_min = self.dm.time_from_start(prev_lat, prev_lon, poi) \
                         if prev_lat == self.start_lat and prev_lon == self.start_lon \
                         else self.dm.time(
                             self._nearest_poi(prev_lat, prev_lon), poi
                         )

            # Distanza incrementale
            from .distance import haversine_km
            dist_km = haversine_km(prev_lat, prev_lon, poi.lat, poi.lon) * 1.3
            total_dist_km += dist_km

            arrival = time_now + travel_min

            # Controlla se arrivo dopo la chiusura → infeasible
            if arrival > poi.time_window.close:
                feasible = False
                # La riparazione dovrebbe impedire questo, ma tracciamo comunque
                wait = 0
            else:
                # Attesa se si arriva prima dell'apertura
                wait    = max(0, poi.time_window.open - arrival)
                arrival = max(arrival, poi.time_window.open)

            departure = arrival + poi.visit_duration
            time_now  = departure
            prev_lat  = poi.lat
            prev_lon  = poi.lon

            schedule.stops.append(ScheduledStop(
                poi=poi, arrival=arrival, departure=departure, wait=wait
            ))

        end_time = self.start_time + self.budget
        schedule.total_time     = time_now - self.start_time
        schedule.total_distance = round(total_dist_km, 2)
        schedule.is_feasible    = feasible and (time_now <= end_time)

        individual._schedule = schedule
        return schedule

    def evaluate(self, individual: Individual) -> FitnessScore:
        """Calcola e assegna la fitness all'individuo."""
        schedule = self.decode(individual)
        end_time = self.start_time + self.budget

        total_score = sum(stop.poi.score for stop in schedule.stops)
        time_over   = max(0, (self.start_time + schedule.total_time) - end_time)

        # Normalizzazione approssimativa (da calibrare sul dataset)
        n_pois      = len(self.dm.pois)
        max_score   = n_pois * 1.0    # se tutti i PoI avessero score=1
        max_dist    = 30.0            # km — stima per un tour cittadino
        max_time    = self.budget

        norm_score = total_score / max_score if max_score > 0 else 0
        norm_dist  = schedule.total_distance / max_dist
        norm_time  = min(schedule.total_time, self.budget) / max_time

        time_penalty = (time_over / 60) * self.penalty  # penalità per ora di sforamento

        scalar = (
            self.w_score * norm_score
            - self.w_dist * norm_dist
            - self.w_time * norm_time
            - time_penalty
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

    def _nearest_poi(self, lat: float, lon: float):
        """Trova il PoI nella matrice più vicino a una coppia lat/lon."""
        from .distance import haversine_km
        return min(self.dm.pois, key=lambda p: haversine_km(lat, lon, p.lat, p.lon))