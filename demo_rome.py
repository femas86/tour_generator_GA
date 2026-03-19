"""
demo_rome.py — Demo aggiornato: TouristProfile + fix ristoranti + tempi transit realistici.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import PoI, PoICategory, TimeWindow
from core.distance import DistanceMatrix
from core.profile import (
    profile_cultural_walker, profile_foodie_transit,
    profile_family_mixed, profile_art_lover_car,
    TouristProfile, TransportMode, MobilityLevel,
)
from solver import NSGA2Solver, SolverConfig

ROME_POIS = [
    # Monumenti e luoghi
    PoI("colosseo",   "Colosseo",             41.8902,12.4922,0.98,120,TimeWindow(540,1110),PoICategory.MONUMENT,  ["antico","unesco"]),
    PoI("foro",       "Foro Romano",           41.8925,12.4853,0.90, 90,TimeWindow(540,1110),PoICategory.MONUMENT,  ["antico"]),
    PoI("vaticano",   "Musei Vaticani",        41.9065,12.4534,0.97,180,TimeWindow(540,1080),PoICategory.MUSEUM,    ["arte","unesco"]),
    PoI("sistina",    "Cappella Sistina",      41.9029,12.4545,0.96, 60,TimeWindow(540,1080),PoICategory.MUSEUM,    ["arte","rinascimento"]),
    PoI("pantheon",   "Pantheon",              41.8986,12.4769,0.93, 60,TimeWindow(540,1140),PoICategory.MONUMENT,  ["antico","architettura"]),
    PoI("trevi",      "Fontana di Trevi",      41.9009,12.4833,0.88, 30,TimeWindow(0,  1440),PoICategory.MONUMENT,  ["barocco","fotogenico"]),
    PoI("spagna",     "Piazza di Spagna",      41.9059,12.4823,0.80, 30,TimeWindow(0,  1440),PoICategory.VIEWPOINT, ["shopping","fotogenico"]),
    PoI("borghese",   "Galleria Borghese",     41.9143,12.4923,0.92,120,TimeWindow(540,1140),PoICategory.MUSEUM,    ["arte","scultura"]),
    PoI("navona",     "Piazza Navona",         41.8992,12.4731,0.85, 45,TimeWindow(0,  1440),PoICategory.VIEWPOINT, ["barocco","fotogenico"]),
    PoI("trastevere", "Trastevere",            41.8897,12.4703,0.82, 90,TimeWindow(600,1380),PoICategory.VIEWPOINT, ["quartiere","fotogenico"]),
    PoI("castel",     "Castel Sant'Angelo",    41.9031,12.4663,0.83, 90,TimeWindow(540,1080),PoICategory.MONUMENT,  ["medievale","panorama"]),
    PoI("aventino",   "Giardino degli Aranci", 41.8837,12.4793,0.75, 40,TimeWindow(480,1200),PoICategory.PARK,      ["panorama","fotogenico"]),
    PoI("terme",      "Terme di Caracalla",    41.8788,12.4924,0.78, 75,TimeWindow(540,1080),PoICategory.MONUMENT,  ["antico"]),
    # Ristoranti: solo per pranzo (TW 12-15) o cena (TW 19-22)
    PoI("rist_rione", "Osteria del Rione",     41.8962,12.4751,0.74, 60,TimeWindow(720, 900),PoICategory.RESTAURANT,["cucina_romana"]),
    PoI("rist_prati", "Trattoria Prati",       41.9042,12.4601,0.70, 75,TimeWindow(720, 900),PoICategory.RESTAURANT,["cucina_romana"]),
    PoI("rist_testac","Testaccio da Mario",    41.8792,12.4770,0.76, 70,TimeWindow(720, 900),PoICategory.RESTAURANT,["offal","cucina_romana"]),
    PoI("rist_cena1", "Ristorante San Pietro", 41.9050,12.4580,0.72, 80,TimeWindow(1140,1320),PoICategory.RESTAURANT,["cucina_romana"]),
    PoI("rist_cena2", "Da Enzo al 29",         41.8891,12.4697,0.78, 90,TimeWindow(1140,1320),PoICategory.RESTAURANT,["cucina_romana","trastevere"]),
    # Bar e caffè: colazione o pausa (TW ampia)
    PoI("bar_greco",  "Caffè Greco",           41.9057,12.4818,0.72, 20,TimeWindow(480,1320),PoICategory.BAR,       ["storico","caffe"]),
    PoI("bar_sant",   "Sant'Eustachio il Caffè",41.8990,12.4752,0.75, 20,TimeWindow(480,1380),PoICategory.BAR,      ["caffe","storico"]),
    PoI("bar_campo",  "Bar del Fico",          41.8968,12.4720,0.65, 25,TimeWindow(600,1380),PoICategory.BAR,       ["aperitivo","vivace"]),
    # Gelaterie: pomeriggio (TW 11-20)
    PoI("gel_fatamorgana","Fatamorgana",       41.8993,12.4729,0.70, 20,TimeWindow(660,1200),PoICategory.GELATERIA, ["artigianale","insolito"]),
    PoI("gel_giolitti",   "Giolitti",          41.9003,12.4765,0.68, 20,TimeWindow(660,1260),PoICategory.GELATERIA, ["storico","classico"]),
    PoI("gel_prati",      "Fatamorgana Prati", 41.9090,12.4626,0.66, 20,TimeWindow(660,1260),PoICategory.GELATERIA, ["artigianale"]),
]


def profile_foodie_transit_updated() -> TouristProfile:
    """Gastronomico con mezzi: pranzo + cena, bar e gelateria nel pomeriggio."""
    return TouristProfile(
        transport_mode     = TransportMode.TRANSIT,
        allowed_categories = ["restaurant", "bar", "gelateria", "monument", "viewpoint", "park"],
        want_lunch         = True,
        want_dinner        = True,
        lunch_time         = 720,   # 12:00
        dinner_time        = 1200,  # 20:00
        meal_window        = 60,
        tag_weights        = {"cucina_romana": 1.6, "offal": 0.5, "vivace": 1.2, "caffe": 1.1},
    )


def run_profile(name, profile, config, dm):
    print(f"\n{'━'*60}")
    print(f"  Profilo: {name}")
    print(f"{'━'*60}")
    print(profile.summary())
    dm.profile = profile
    solver = NSGA2Solver(ROME_POIS, dm, config, profile=profile)

    def cb(gen, pareto, stats):
        if gen % 30 == 0 or gen == 1:
            print(f"  gen {gen:3d} | pareto={stats['pareto_size']:2d} | "
                  f"best={stats['best_scalar']:.4f} | feasible={stats['feasible_pct']:.0f}%")

    front = solver.solve(callback=cb)
    feasible = [x for x in front if x.fitness.is_feasible] or front
    if not feasible:
        print("  Nessuna soluzione trovata.")
        return
    best = max(feasible, key=lambda x: x.fitness.scalar)
    sched = solver.evaluator.decode(best)

    # Conta categorie per verifica
    cats = {}
    for stop in sched.stops:
        k = stop.poi.category.value
        cats[k] = cats.get(k, 0) + 1

    print(f"\n  ★ Tour: {len(best.genes)} PoI | score={best.fitness.total_score:.2f} | "
          f"{best.fitness.total_distance:.1f}km | {best.fitness.total_time}min")
    print(f"     Composizione: {', '.join(f'{v}×{k}' for k,v in sorted(cats.items()))}")
    print()
    if sched:
        print(sched.summary())


def main():
    print("Costruzione matrice distanze...")
    dm = DistanceMatrix(ROME_POIS)
    dm.build()

    config = SolverConfig(
        pop_size         = 60,
        max_generations  = 200,
        budget           = 660,    # 11 ore (09:30–20:30)
        start_time       = 570,    # 09:30
        start_lat        = 41.896,
        start_lon        = 12.484,
        stagnation_limit = 25,
    )

    profiles = [
        ("Gastronomico con mezzi (aggiornato)", profile_foodie_transit_updated()),
        ("Culturale a piedi",                   profile_cultural_walker()),
        ("Gastronomico con mezzi (standard)", profile_foodie_transit()),
        ("Family: misto",                      profile_family_mixed()),
        ("Art Lover: con auto",                 profile_art_lover_car()),
        ("Custom: solo viste, no pasti",         TouristProfile(
            transport_mode=TransportMode.WALK,
            allowed_categories=["monument","viewpoint","park","bar","gelateria"],
            want_lunch=False, want_dinner=False,
            tag_weights={"fotogenico":1.5,"panorama":1.4,"caffe":1.1},
        )),
    ]

    for name, profile in profiles:
        run_profile(name, profile, config, dm)

    print(f"\n{'━'*60}\n  Completato.\n")


if __name__ == "__main__":
    main()