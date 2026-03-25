"""
solver.py — Entry point principale. Ciclo evolutivo NSGA-II per TOP-TW.
Uso:
    python -m tour_ga.solver --city rome --budget 480 --generations 200
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import Callable

from core.models import Individual
from core.distance import DistanceMatrix
from core.fitness import FitnessEvaluator
from ga.operators import (
    tournament_select, order_crossover, poi_aware_crossover, mutate
)
from ga.repair import RepairEngine
from ga.seeding import GreedySeeder
import config


@dataclass
class SolverConfig:
    pop_size:         int   = config.GA_POP_SIZE
    max_generations:  int   = config.GA_MAX_GENERATIONS
    cx_prob:          float = config.GA_CX_PROB
    mut_prob:         float = config.GA_MUT_PROB
    tournament_k:     int   = config.GA_TOURNAMENT_K
    stagnation_limit: int   = config.GA_STAGNATION_LIMIT
    start_time:       int   = config.DEFAULT_START_TIME
    budget:           int   = config.DEFAULT_BUDGET
    start_lat:        float = config.DEFAULT_START_LAT
    start_lon:        float = config.DEFAULT_START_LON
    w_score:          float = config.W_SCORE
    w_dist:           float = config.W_DIST
    w_time:           float = config.W_TIME
    max_wait_min:     int   = config.GA_MAX_WAIT_MIN
    ox_crossover_prob: float = config.GA_OX_CROSSOVER_PROB


class NSGA2Solver:
    """
    Implementazione NSGA-II adattata al problema TOP-TW.
    Riceve un TouristProfile e lo propaga a tutti i componenti.
    """

    def __init__(self, pois, dm: DistanceMatrix, config: SolverConfig, profile=None):
        from core.profile import TouristProfile # import locale per evitare dipendenze circolari
        self.pois    = pois
        self.dm      = dm
        self.config  = config
        self.profile = profile or TouristProfile()   # default: turista generico a piedi

        # Inietta il profilo nella matrice distanze (per la velocità)
        self.dm.profile = self.profile

        self.repair = RepairEngine(
            dm=dm,
            profile=self.profile,
            all_pois=pois,
            start_time=config.start_time,
            budget=config.budget,
            start_lat=config.start_lat,
            start_lon=config.start_lon,
            max_wait_min=config.max_wait_min,
        )
        self.evaluator = FitnessEvaluator(
            dist_matrix=dm,
            profile=self.profile,
            start_time=config.start_time,
            budget=config.budget,
            start_lat=config.start_lat,
            start_lon=config.start_lon,
            w_score=config.w_score,
            w_dist=config.w_dist,
            w_time=config.w_time,
        )
        self.seeder = GreedySeeder(
            pois=pois,
            dm=dm,
            repair=self.repair,
            profile=self.profile,
            start_time=config.start_time,
            budget=config.budget,
            start_lat=config.start_lat,
            start_lon=config.start_lon,
        )

        self.history: list[dict] = []  # statistiche per generazione

    def solve(self, callback: Callable | None = None) -> list[Individual]:
        """
        Esegue il ciclo NSGA-II e restituisce il fronte di Pareto finale.
        callback(gen, pareto_front, stats) viene chiamata ogni generazione.
        """
        cfg = self.config
        t0  = time.perf_counter()

        # --- Inizializzazione ---
        population = self.seeder.build_population(cfg.pop_size)
        population = self._evaluate_all(population)
        population = self._nsga2_select(population + [], cfg.pop_size)

        best_scalar   = max(ind.fitness.scalar for ind in population)
        stagnation    = 0

        for gen in range(cfg.max_generations):
            # --- Generazione figli ---
            offspring = []
            while len(offspring) < cfg.pop_size:
                p1 = tournament_select(population, cfg.tournament_k)
                p2 = tournament_select(population, cfg.tournament_k)

                # Alterna tra OX e PoI-aware crossover
                if random.random() < cfg.cx_prob:
                    if random.random() < cfg.ox_crossover_prob:
                        c1, c2 = order_crossover(p1, p2)
                    else:
                        c1, c2 = poi_aware_crossover(p1, p2)
                else:
                    c1, c2 = p1.clone(), p2.clone()

                c1 = mutate(c1, self.seeder.allowed_pois, cfg.mut_prob)
                c2 = mutate(c2, self.seeder.allowed_pois, cfg.mut_prob)

                # Riparazione obbligatoria dopo ogni operatore
                c1 = self.repair.repair(c1)
                c2 = self.repair.repair(c2)

                offspring.extend([c1, c2])

            # --- Valutazione e selezione NSGA-II ---
            offspring  = self._evaluate_all(offspring)
            combined   = population + offspring
            population = self._nsga2_select(combined, cfg.pop_size)

            # --- Statistiche e criterio di stop ---
            pareto    = [ind for ind in population if ind.fitness.rank == 1]
            new_best  = max(ind.fitness.scalar for ind in population)

            stats = {
                "gen":            gen + 1,
                "pareto_size":    len(pareto),
                "best_scalar":    round(new_best, 4),
                "avg_score":      round(
                    sum(ind.fitness.total_score for ind in population) / len(population), 3
                ),
                "feasible_pct":   round(
                    sum(1 for ind in population if ind.fitness.is_feasible) / len(population) * 100, 1
                ),
                "elapsed_s":      round(time.perf_counter() - t0, 2),
            }
            self.history.append(stats)

            if callback:
                callback(gen + 1, pareto, stats)

            if new_best > best_scalar + 1e-6:
                best_scalar = new_best
                stagnation  = 0
            else:
                stagnation += 1

            if stagnation >= cfg.stagnation_limit:
                print(f"  Early stop a gen {gen+1}: stagnazione per {stagnation} generazioni.")
                break

        pareto_front = [ind for ind in population if ind.fitness.rank == 1]
        return sorted(pareto_front, key=lambda x: -x.fitness.total_score)

    # ------------------------------------------------------------------
    # NSGA-II core: fast non-dominated sort + crowding distance
    # ------------------------------------------------------------------

    def _evaluate_all(self, pop: list[Individual]) -> list[Individual]:
        for ind in pop:
            self.evaluator.evaluate(ind)
        return pop

    def _nsga2_select(
        self, combined: list[Individual], target_size: int
    ) -> list[Individual]:
        """Selezione NSGA-II: ranking Pareto + crowding distance."""
        fronts = self._fast_non_dominated_sort(combined)

        next_pop: list[Individual] = []
        for front in fronts:
            if len(next_pop) + len(front) <= target_size:
                next_pop.extend(front)
            else:
                self._assign_crowding_distance(front)
                front.sort(key=lambda x: x.fitness.crowd, reverse=True)
                next_pop.extend(front[:target_size - len(next_pop)])
                break

        return next_pop

    def _fast_non_dominated_sort(
        self, pop: list[Individual]
    ) -> list[list[Individual]]:
        """
        Algoritmo NSGA-II O(MN²) per la costruzione dei fronti.
        M = numero obiettivi, N = dimensione popolazione.
        """
        n = len(pop)
        dom_count  = [0] * n          # quanti individui dominano i
        dom_set    = [[] for _ in range(n)]  # individui dominati da i
        fronts     = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                fi, fj = pop[i].fitness, pop[j].fitness
                if fi.dominates(fj):
                    dom_set[i].append(j)
                elif fj.dominates(fi):
                    dom_count[i] += 1
            if dom_count[i] == 0:
                pop[i].fitness.rank = 1
                fronts[0].append(pop[i])

        rank = 1
        current_front = fronts[0]
        while current_front:
            next_front = []
            for ind in current_front:
                idx_i = pop.index(ind)
                for idx_j in dom_set[idx_i]:
                    dom_count[idx_j] -= 1
                    if dom_count[idx_j] == 0:
                        pop[idx_j].fitness.rank = rank + 1
                        next_front.append(pop[idx_j])
            rank += 1
            fronts.append(next_front)
            current_front = next_front

        return [f for f in fronts if f]

    def _assign_crowding_distance(self, front: list[Individual]):
        """Calcola la crowding distance per il front dato."""
        n = len(front)
        if n == 0:
            return
        for ind in front:
            ind.fitness.crowd = 0.0

        objectives = [
            lambda x:  x.fitness.total_score,      # massimizza
            lambda x: -x.fitness.total_distance,   # minimizza → negativo
            lambda x: -x.fitness.total_time,        # minimizza → negativo
        ]
        for obj_fn in objectives:
            sorted_f = sorted(front, key=obj_fn)
            sorted_f[0].fitness.crowd  = float('inf')
            sorted_f[-1].fitness.crowd = float('inf')
            f_min = obj_fn(sorted_f[0])
            f_max = obj_fn(sorted_f[-1])
            f_range = f_max - f_min or 1e-9
            for i in range(1, n - 1):
                sorted_f[i].fitness.crowd += (
                    obj_fn(sorted_f[i+1]) - obj_fn(sorted_f[i-1])
                ) / f_range