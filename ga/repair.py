"""
ga/repair.py — Motore di riparazione genetica con profilo turista.
Garantisce:
  1. Solo PoI di categorie ammesse dal profilo
  2. Nessuna time window violata
  3. Budget rispettato
  4. Slot pasto presente se richiesto dal profilo
"""
from __future__ import annotations
import random
from core.models import Individual, PoI, PoICategory
from core.distance import DistanceMatrix, haversine_km
from core.profile import TouristProfile


class RepairEngine:

    def __init__(
        self,
        dm:          DistanceMatrix,
        profile:     TouristProfile,
        all_pois:    list[PoI],
        start_time:  int,
        budget:      int,
        start_lat:   float,
        start_lon:   float,
        max_wait_min: int = 30,   # attesa massima tollerata per un singolo PoI (minuti)
    ):
        self.dm           = dm
        self.profile      = profile
        self.all_pois     = all_pois
        self.start_time   = start_time
        self.budget       = budget
        self.start_lat    = start_lat
        self.start_lon    = start_lon
        self.max_wait_min = max_wait_min

    def repair(self, individual: Individual) -> Individual:
        """Pipeline: categoria → EDF → TW → budget → cap ristoranti → cap snack → pasto."""
        individual.invalidate_cache()
        individual = self._filter_allowed_categories(individual)
        individual = self._sort_by_earliest_deadline(individual)
        individual = self.repair_time_windows(individual)
        individual = self.repair_budget(individual)
        individual = self._cap_restaurants(individual)
        individual = self._cap_snacks(individual)
        individual = self._ensure_meal_slots(individual)
        individual.invalidate_cache()
        return individual

    # ------------------------------------------------------------------
    # 1. Rimuove PoI di categorie non ammesse dal profilo
    # ------------------------------------------------------------------

    def _filter_allowed_categories(self, individual: Individual) -> Individual:
        individual.genes = [
            p for p in individual.genes
            if self.profile.allows_category(p.category.value)
        ]
        return individual

    # ------------------------------------------------------------------
    # 2. Ordina per Earliest Deadline First
    # ------------------------------------------------------------------

    def _sort_by_earliest_deadline(self, individual: Individual) -> Individual:
        """
        Ordina per earliest OPEN time: visita prima i PoI che aprono prima.
        Evita che i ristoranti (open=720) finiscano in cima creando attese enormi.
        """
        individual.genes.sort(key=lambda p: p.time_window.open)
        return individual

    # ------------------------------------------------------------------
    # 3. Rimuove PoI con TW violata
    # ------------------------------------------------------------------

    def repair_time_windows(self, individual: Individual) -> Individual:
        """
        Simula il tour e rimuove i PoI che causano problemi:
          - arrivo dopo la chiusura (infeasible)
          - attesa all'apertura superiore a max_wait_min (tour non realistico)
        """
        valid    = []
        time_now = self.start_time
        prev_lat = self.start_lat
        prev_lon = self.start_lon

        for poi in individual.genes:
            travel  = self._travel_min(prev_lat, prev_lon, poi)
            arrival = time_now + travel

            if arrival > poi.time_window.close:
                continue   # fuori orario

            wait = max(0, poi.time_window.open - arrival)
            if wait > self.max_wait_min:
                continue   # attesa inaccettabile: rimuovi

            arrival   = arrival + wait
            departure = arrival + poi.visit_duration

            valid.append(poi)
            time_now = departure
            prev_lat = poi.lat
            prev_lon = poi.lon

        individual.genes = valid
        return individual

    # ------------------------------------------------------------------
    # 4. Rimuove PoI finché budget rispettato (minor score/durata prima)
    # ------------------------------------------------------------------

    def repair_budget(self, individual: Individual) -> Individual:
        while True:
            total = self._simulate_total_time(individual.genes)
            if total <= self.budget or not individual.genes:
                break
            worst = min(
                individual.genes,
                key=lambda p: p.score / (p.visit_duration + 1e-9)
            )
            individual.genes.remove(worst)
        return individual

    # ------------------------------------------------------------------
    # 5. Limita i ristoranti al numero di slot pasto richiesti
    # ------------------------------------------------------------------

    def _cap_restaurants(self, individual: Individual) -> Individual:
        """
        Rimuove i ristoranti in eccesso rispetto agli slot pasto del profilo.

        Logica:
          - Conta quanti slot pasto il profilo richiede (0, 1, 2)
          - Se ci sono più ristoranti nel tour di quanti ne servono,
            mantieni solo i migliori N per score (uno per slot),
            rispettando l'orario di apertura del slot corrispondente.
          - BAR e GELATERIA non vengono contati come ristoranti e
            non subiscono questo cappuccio.
        """
        n_slots = len(self.profile.needs_meal_slot())

        restaurants_in_tour = [
            p for p in individual.genes
            if p.category == PoICategory.RESTAURANT
        ]

        if len(restaurants_in_tour) <= n_slots:
            return individual  # già OK

        # Tieni solo i migliori N ristoranti per score
        restaurants_in_tour.sort(key=lambda p: p.score, reverse=True)
        keep    = set(p.id for p in restaurants_in_tour[:n_slots])
        remove  = set(p.id for p in restaurants_in_tour[n_slots:])

        individual.genes = [
            p for p in individual.genes
            if p.id not in remove
        ]
        return individual

    def _cap_snacks(self, individual: Individual) -> Individual:
        """
        Limita le soste snack (BAR, GELATERIA) ai massimi definiti nel profilo.
        Mantiene le soste con score più alto, rimuove le eccedenti.
        """
        for cat, max_stops in [
            (PoICategory.BAR,       self.profile.max_bar_stops),
            (PoICategory.GELATERIA, self.profile.max_gelateria_stops),
        ]:
            in_tour = [p for p in individual.genes if p.category == cat]
            if len(in_tour) <= max_stops:
                continue
            in_tour.sort(key=lambda p: p.score, reverse=True)
            remove = {p.id for p in in_tour[max_stops:]}
            individual.genes = [p for p in individual.genes if p.id not in remove]
        return individual

    # ------------------------------------------------------------------
    # 6. Garantisce slot pasto se richiesto dal profilo
    # ------------------------------------------------------------------

    def _ensure_meal_slots(self, individual: Individual) -> Individual:
        """
        Per ogni slot pasto richiesto dal profilo, garantisce un ristorante.

        Strategia:
          1. Prova ad AGGIUNGERE il ristorante senza sforare il budget.
          2. Se non c'è spazio, prova a SOSTITUIRE il PoI con il peggior
             rapporto score/durata con il ristorante, se ciò libera abbastanza
             tempo da farci stare il pasto.
        """
        meal_slots = self.profile.needs_meal_slot()
        if not meal_slots:
            return individual

        restaurants = [
            p for p in self.all_pois
            if p.category == PoICategory.RESTAURANT
            and self.profile.allows_category(p.category.value)
            and p not in individual.genes
        ]
        if not restaurants:
            return individual

        for (slot_open, slot_close) in meal_slots:
            schedule = self._simulate_schedule(individual.genes)
            already_covered = any(
                stop["poi"].category == PoICategory.RESTAURANT
                and slot_open <= stop["arrival"] <= slot_close
                for stop in schedule
            )
            if already_covered:
                continue

            inserted = False

            # --- Tentativo 1: inserimento diretto ---
            for rest in sorted(restaurants, key=lambda r: r.score, reverse=True):
                if rest in individual.genes:
                    continue
                for pos in range(len(individual.genes) + 1):
                    test_genes = individual.genes[:pos] + [rest] + individual.genes[pos:]
                    test_sched = self._simulate_schedule(test_genes, max_wait_override=45)
                    if not test_sched:
                        continue
                    rest_stop = next((s for s in test_sched if s["poi"].id == rest.id), None)
                    if rest_stop is None:
                        continue
                    arrival = rest_stop["arrival"]
                    total_t = test_sched[-1]["departure"] - self.start_time
                    if slot_open <= arrival <= slot_close and total_t <= self.budget:
                        individual.genes.insert(pos, rest)
                        inserted = True
                        break
                if inserted:
                    break

            if inserted:
                continue

            # --- Tentativo 2: sostituzione del PoI meno prezioso ---
            # Ordina i candidati alla rimozione per minor valore (escludi ristoranti già presenti)
            removable = [
                p for p in individual.genes
                if p.category not in (PoICategory.RESTAURANT, PoICategory.BAR, PoICategory.GELATERIA)
            ]
            if not removable:
                continue
            removable.sort(key=lambda p: p.score / (p.visit_duration + 1e-9))

            for victim in removable:
                for rest in sorted(restaurants, key=lambda r: r.score, reverse=True):
                    if rest in individual.genes:
                        continue
                    test_genes = [r if r.id != victim.id else rest for r in individual.genes]
                    test_sched = self._simulate_schedule(test_genes, max_wait_override=45)
                    if not test_sched:
                        continue
                    rest_stop = next((s for s in test_sched if s["poi"].id == rest.id), None)
                    if rest_stop is None:
                        continue
                    arrival = rest_stop["arrival"]
                    total_t = test_sched[-1]["departure"] - self.start_time
                    if slot_open <= arrival <= slot_close and total_t <= self.budget:
                        idx = individual.genes.index(victim)
                        individual.genes[idx] = rest
                        inserted = True
                        break
                if inserted:
                    break

        individual.invalidate_cache()
        return individual

    # ------------------------------------------------------------------
    # Helper interni
    # ------------------------------------------------------------------

    def _simulate_schedule(self, genes: list[PoI], max_wait_override: int | None = None) -> list[dict]:
        """
        Simula lo schedule. Salta i PoI con TW violata o attesa eccessiva.
        max_wait_override: se specificato, usa questa soglia invece di self.max_wait_min.
        """
        max_wait = max_wait_override if max_wait_override is not None else self.max_wait_min
        stops    = []
        time_now = self.start_time
        prev_lat = self.start_lat
        prev_lon = self.start_lon

        for poi in genes:
            travel  = self._travel_min(prev_lat, prev_lon, poi)
            arrival = time_now + travel
            if arrival > poi.time_window.close:
                continue
            wait = max(0, poi.time_window.open - arrival)
            if wait > max_wait:
                continue
            arrival   = arrival + wait
            departure = arrival + poi.visit_duration
            stops.append({"poi": poi, "arrival": arrival, "departure": departure, "wait": wait})
            time_now = departure
            prev_lat = poi.lat
            prev_lon = poi.lon

        return stops

    def _simulate_total_time(self, genes: list[PoI]) -> int:
        sched = self._simulate_schedule(genes)
        if not sched:
            return self.budget + 1   # infeasible → forza rimozione
        return sched[-1]["departure"] - self.start_time

    def _travel_min(self, lat: float, lon: float, to: PoI) -> int:
        km = haversine_km(lat, lon, to.lat, to.lon) * 1.3
        return self.profile.travel_time_min(km)