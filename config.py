"""
config.py — Configurazione centralizzata del progetto tour_ga.

Tutte le costanti "magiche" del progetto sono raccolte qui.
I moduli importano da questo file invece di avere valori hardcoded.

Struttura:
  TRANSPORT  — velocità, overhead, soglie di modalità
  FITNESS    — pesi obiettivi, penalità, normalizzazione
  REPAIR     — vincoli di riparazione genetica
  SEEDING    — parametri costruzione popolazione iniziale
  GA         — default algoritmo evolutivo
  VISUALIZER — colori e stili per la mappa HTML
"""
from __future__ import annotations

# ============================================================
# TRANSPORT — Modello di percorrenza realistico
# ============================================================

# Fattore correttivo Haversine → percorso stradale reale
# 1.0 = linea d'aria, 1.3 = stima tipica percorso urbano
ROUTE_DETOUR_FACTOR: float = 1.3

# Soglia sotto la quale, anche in modalità MIXED, si usa v_walk
# (600 m → preferisce a piedi rispetto al mezzo)
MIXED_THRESHOLD_M: int = 600

# Soglia sotto la quale non conviene prendere il mezzo in TRANSIT
# (prendere bus/metro per <400m è più lento del cammino)
TRANSIT_WALK_THRESHOLD_KM: float = 0.40

# Overhead fisso per ogni tratta in mezzo pubblico (minuti)
# Comprende: cammino alla fermata + attesa mezzo + cammino dalla fermata
# Roma: bus ~8-12 min freq., metro ~4-5 min freq. → media 10 min
TRANSIT_OVERHEAD_MIN: int = 10

# Overhead fisso per ogni tratta in auto/taxi (minuti)
# Comprende: ricerca parcheggio + cammino dal parcheggio
CAR_OVERHEAD_MIN: int = 5

# Velocità medie di percorrenza in km/h (escluso overhead)
# Chiave: (TransportMode.value, MobilityLevel.value)
SPEED_KMH: dict[tuple[str, str], float] = {
    ("walk",    "normal"):  4.5,
    ("walk",    "limited"): 3.0,
    ("car",     "normal"):  25.0,
    ("car",     "limited"): 25.0,
    ("transit", "normal"):  20.0,   # metro Roma in media ~20 km/h
    ("transit", "limited"): 16.0,
    ("mixed",   "normal"):  4.5,    # segmenti brevi → v_walk
    ("mixed",   "limited"): 3.0,
}

# Velocità per segmenti lunghi in modalità MIXED (oltre MIXED_THRESHOLD_M)
MIXED_LONG_SPEED_KMH: dict[str, float] = {
    "normal":  20.0,
    "limited": 16.0,
}

# ============================================================
# FITNESS — Funzione di valutazione multi-obiettivo
# ============================================================

# Pesi default per la funzione scalare aggregata
W_SCORE: float = 0.50   # peso obiettivo score (da massimizzare)
W_DIST:  float = 0.20   # peso obiettivo distanza (da minimizzare)
W_TIME:  float = 0.30   # peso penalità tempo (non usato nello scalare diretto)

# Penalità per sforamento budget (per ora di sforamento)
PENALTY_BUDGET_OVERRUN: float = 50.0

# Penalità scalare per slot pasto non coperto
# Applicata solo se "restaurant" è nelle categorie ammesse dal profilo
PENALTY_MEAL_MISSING: float = 0.25

# Soglia di attesa cumulata (minuti) sotto cui non si penalizza
# Attese brevi (es. 3 min prima dell'apertura) sono accettabili
WAIT_PENALTY_THRESHOLD_MIN: int = 5

# Fattore di penalità per i minuti di attesa eccedenti la soglia
# (per ora di attesa cumulata oltre la soglia)
WAIT_PENALTY_FACTOR: float = 10.0

# Cap moltiplicativo per effective_score con boost da tag_weights
# Evita che tag boost portino lo score molto sopra 1.0
SCORE_BOOST_CAP: float = 1.5

# Distanza massima di normalizzazione per la fitness (km)
# Dipende dalla modalità di trasporto
MAX_DIST_WALK_KM:    float = 15.0
MAX_DIST_TRANSIT_KM: float = 50.0
MAX_DIST_CAR_KM:     float = 80.0

# Minuti di visita extra per ogni membro del gruppo oltre il primo
GROUP_VISIT_OVERHEAD_PER_PERSON: int = 5

# Fitness utilization bonus
FITNESS_UTILIZATION_BONUS_FACTOR : float = 0.3

# ============================================================
# REPAIR — Vincoli del motore di riparazione genetica
# ============================================================

# Attesa massima tollerata per qualsiasi PoI (minuti)
# PoI che richiederebbero un'attesa maggiore vengono rimossi dal tour
MAX_WAIT_MIN: int = 30

# Tolleranza di attesa speciale per l'inserimento di ristoranti
# nei slot pasto: arrivare poco prima dell'apertura è comportamento normale
MEAL_SLOT_WAIT_OVERRIDE_MIN: int = 45

# Fraction della lunghezza del tour usata nell'ordinamento per
# spostare i ristoranti nella posizione temporalmente corretta
# (non una costante numerica, ma un commento di design)

# ============================================================
# SEEDING — Costruzione della popolazione iniziale
# ============================================================

# Frazione di individui costruiti con greedy deterministico
SEED_GREEDY_FRACTION: float = 0.20

# Frazione di individui costruiti con α-greedy perturbato
SEED_PERTURBED_FRACTION: float = 0.20

# Il restante (1 - greedy - perturbed) viene costruito casualmente e riparato

# Range del parametro alpha per l'α-greedy perturbato
# alpha=0 → greedy puro; alpha=0.5 → semi-casuale (GRASP-like)
SEED_ALPHA_MIN: float = 0.15
SEED_ALPHA_MAX: float = 0.50   # alpha_min + 0.35

# Dimensione della Restricted Candidate List come frazione dei candidati
# (top RCL_FRACTION vengono estratti casualmente invece del migliore assoluto)
RCL_FRACTION: float = 0.20

# ============================================================
# PASTI - Valori di default
# ============================================================

# --- Preferenze pasti ---
WANT_LUNCH:             bool = True
WANT_DINNER:            bool = False
LUNCH_TIME:             int  = 720
DINNER_TIME:            int  = 1140
MEAL_WINDOW:            int  = 60
MAX_BAR_STOPS:          int = 1
MAX_GELATERIA_STOPS:    int = 1
MEAL_RESERVE_MIN:       int  = 90
EVENING_THRESHOLD:      int  = 1140   # 19:00

# ============================================================
# GA — Default dell'algoritmo evolutivo (usati da SolverConfig)
# ============================================================

GA_POP_SIZE:         int   = 80
GA_MAX_GENERATIONS:  int   = 300
GA_CX_PROB:          float = 0.85    # probabilità di crossover
GA_MUT_PROB:         float = 0.20    # probabilità di mutazione
GA_TOURNAMENT_K:     int   = 3       # candidati per torneo
GA_STAGNATION_LIMIT: int   = 50      # generazioni senza miglioramento → stop
GA_MAX_WAIT_MIN:     int   = MAX_WAIT_MIN   # propagato al RepairEngine
GA_OX_CROSSOVER_PROB: float = 0.60
# Probabilità di usare Order Crossover (OX) vs PoI-aware crossover,
# condizionata all'aver già deciso di fare crossover (cx_prob).
# OX è più conservativo (preserva ordine globale),
# PoI-aware è più espolorativo (scambia blocchi per categoria).
# Un mix 60/40 bilancia convergenza ed esplorazione tematica.
# Orario di partenza e budget default (minuti dalla mezzanotte)

DEFAULT_START_TIME: int = 540    # 09:00
DEFAULT_BUDGET:     int = 480    # 8 ore

# Coordinate default (Roma, centro storico)
DEFAULT_START_LAT: float = 41.8960
DEFAULT_START_LON: float = 12.4840

# ============================================================
# VISUALIZER — Stili per la mappa HTML interattiva
# ============================================================

# Colori hex per categoria PoI sulla mappa
# Basati sui ramp del design system del progetto
CATEGORY_COLORS: dict[str, str] = {
    "museum":     "#7F77DD",   # purple-400
    "monument":   "#378ADD",   # blue-400
    "restaurant": "#D85A30",   # coral-400
    "bar":        "#BA7517",   # amber-400
    "gelateria":  "#D4537E",   # pink-400
    "park":       "#639922",   # green-400
    "viewpoint":  "#1D9E75",   # teal-400
}

# Colori speciali per elementi della mappa
MAP_ROUTE_COLOR:     str = "#378ADD"   # polyline del percorso
MAP_START_COLOR:     str = "#1D9E75"   # marker punto di partenza
MAP_ARROW_COLOR:     str = "#5F5E5A"   # frecce di direzione

# Opacità della polyline del percorso (0.0–1.0)
MAP_ROUTE_OPACITY: float = 0.75

# Spessore della polyline in pixel
MAP_ROUTE_WEIGHT: int = 4

# Raggio dei cerchi marker in pixel
MAP_MARKER_RADIUS: int = 10

# Zoom default sulla mappa al caricamento
MAP_ZOOM_DEFAULT: int = 14

# URL tiles OpenStreetMap (nessuna API key richiesta)
MAP_TILE_URL: str = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
MAP_TILE_ATTRIBUTION: str = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)