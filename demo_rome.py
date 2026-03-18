"""
demo_rome.py — Demo completo: genera tour per Roma con PoI reali.
Esegui con: python demo_rome.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from core.models import PoI, PoICategory, TimeWindow
from core.distance import DistanceMatrix
from solver import NSGA2Solver, SolverConfig


# ---------------------------------------------------------------------------
# Dataset PoI – Roma (coordinate reali, score e durate indicativi)
# ---------------------------------------------------------------------------
ROME_POIS = [
    PoI("colosseo",    "Colosseo",             41.8902, 12.4922, 0.98, 120, TimeWindow(540, 1110),  PoICategory.MONUMENT,  ["antico", "unesco"]),
    PoI("foro",        "Foro Romano",           41.8925, 12.4853, 0.90, 90,  TimeWindow(540, 1110),  PoICategory.MONUMENT,  ["antico"]),
    PoI("vaticano",    "Musei Vaticani",        41.9065, 12.4534, 0.97, 180, TimeWindow(540, 1080),  PoICategory.MUSEUM,    ["arte", "unesco"]),
    PoI("sistina",     "Cappella Sistina",      41.9029, 12.4545, 0.96, 60,  TimeWindow(540, 1080),  PoICategory.MUSEUM,    ["arte", "rinascimento"]),
    PoI("pantheon",    "Pantheon",              41.8986, 12.4769, 0.93, 60,  TimeWindow(540, 1140),  PoICategory.MONUMENT,  ["antico", "architettura"]),
    PoI("trevi",       "Fontana di Trevi",      41.9009, 12.4833, 0.88, 30,  TimeWindow(0,   1440),  PoICategory.MONUMENT,  ["barocco", "fotogenico"]),
    PoI("spagna",      "Piazza di Spagna",      41.9059, 12.4823, 0.80, 30,  TimeWindow(0,   1440),  PoICategory.VIEWPOINT, ["shopping", "fotogenico"]),
    PoI("borghese",    "Galleria Borghese",     41.9143, 12.4923, 0.92, 120, TimeWindow(540, 1140),  PoICategory.MUSEUM,    ["arte", "scultura"]),
    PoI("navona",      "Piazza Navona",         41.8992, 12.4731, 0.85, 45,  TimeWindow(0,   1440),  PoICategory.VIEWPOINT, ["barocco", "fotogenico"]),
    PoI("trastevere",  "Trastevere",            41.8897, 12.4703, 0.82, 90,  TimeWindow(600, 1380),  PoICategory.VIEWPOINT, ["quartiere", "fotogenico"]),
    PoI("terme",       "Terme di Caracalla",    41.8788, 12.4924, 0.78, 75,  TimeWindow(540, 1080),  PoICategory.MONUMENT,  ["antico"]),
    PoI("castel",      "Castel Sant'Angelo",    41.9031, 12.4663, 0.83, 90,  TimeWindow(540, 1080),  PoICategory.MONUMENT,  ["medievale", "panorama"]),
    PoI("campo_fiori", "Campo de' Fiori",       41.8955, 12.4722, 0.72, 30,  TimeWindow(480, 840),   PoICategory.VIEWPOINT, ["mercato", "vivace"]),
    # Ristoranti (time window = orario pranzo o cena)
    PoI("rist_rione",  "Osteria del Rione",     41.8962, 12.4751, 0.74, 60,  TimeWindow(720, 870),   PoICategory.RESTAURANT, ["cucina_romana"]),
    PoI("rist_prati",  "Trattoria Prati",       41.9042, 12.4601, 0.70, 75,  TimeWindow(720, 870),   PoICategory.RESTAURANT, ["cucina_romana"]),
    PoI("rist_testac", "Testaccio da Mario",    41.8792, 12.4770, 0.76, 70,  TimeWindow(720, 870),   PoICategory.RESTAURANT, ["offal", "cucina_romana"]),
]


def print_tour(individual, idx=1):
    schedule = individual._schedule
    print(f"\n{'='*55}")
    print(f"  Tour #{idx}  |  {len(individual.genes)} PoI  |  "
          f"Score: {individual.fitness.total_score:.2f}  |  "
          f"Dist: {individual.fitness.total_distance:.1f} km  |  "
          f"Tempo: {individual.fitness.total_time} min")
    print(f"{'='*55}")
    if schedule:
        print(schedule.summary())


def main():
    print("Costruzione matrice distanze...")
    dm = DistanceMatrix(ROME_POIS, mode="walk")
    dm.build()

    config = SolverConfig(
        pop_size        = 60,
        max_generations = 150,
        budget          = 480,   # 8 ore
        start_time      = 540,   # 09:00
        start_lat       = 41.8960,
        start_lon       = 12.4840,  # partenza: centro storico
        stagnation_limit = 40,
    )

    solver = NSGA2Solver(ROME_POIS, dm, config)

    print(f"Avvio NSGA-II: pop={config.pop_size}, gen={config.max_generations}\n")

    def progress(gen, pareto, stats):
        if gen % 20 == 0 or gen == 1:
            print(f"  Gen {gen:3d} | Pareto: {stats['pareto_size']:2d} | "
                  f"Best scalar: {stats['best_scalar']:.4f} | "
                  f"Feasible: {stats['feasible_pct']:.0f}% | "
                  f"{stats['elapsed_s']:.1f}s")

    pareto_front = solver.solve(callback=progress)

    print(f"\nFronte di Pareto finale: {len(pareto_front)} soluzioni\n")

    # Mostra le prime 3 soluzioni del fronte
    for i, ind in enumerate(pareto_front[:3], 1):
        schedule = solver.evaluator.decode(ind)
        print_tour(ind, i)

    # Esempio: selezione automatica della soluzione bilanciata
    # (massimizza score pesato con pari importanza sugli obiettivi)
    balanced = max(
        pareto_front,
        key=lambda x: x.fitness.total_score - 0.5 * x.fitness.total_distance / 10
    )
    print(f"\n>> Soluzione bilanciata consigliata: Tour con {len(balanced.genes)} PoI")
    solver.evaluator.decode(balanced)
    print_tour(balanced, idx="★")


if __name__ == "__main__":
    main()