# tour_ga — Generatore di tour turistici con NSGA-II e TOP-TW

Progetto Python per la generazione automatica di tour personalizzati in città (Roma come caso di studio) tramite un algoritmo genetico multi-obiettivo. Il problema è modellato come **Team Orienteering Problem with Time Windows (TOP-TW)** e risolto con una variante di **NSGA-II** (Non-dominated Sorting Genetic Algorithm II, Deb et al., 2002).

---

## Indice

1. [Il problema: TOP-TW](#1-il-problema-top-tw)
2. [Fondamenti teorici: NSGA-II](#2-fondamenti-teorici-nsga-ii)
3. [Struttura del progetto](#3-struttura-del-progetto)
4. [Modello dati](#4-modello-dati)
5. [Profilo turista (TouristProfile)](#5-profilo-turista-touristprofile)
6. [Funzione fitness multi-obiettivo](#6-funzione-fitness-multi-obiettivo)
7. [Operatori genetici](#7-operatori-genetici)
8. [Riparazione genetica (Repair Engine)](#8-riparazione-genetica-repair-engine)
9. [Greedy Seeding](#9-greedy-seeding)
10. [Modello di trasporto realistico](#10-modello-di-trasporto-realistico)
11. [Configurazione e avvio](#11-configurazione-e-avvio)
12. [Risultati di esempio](#12-risultati-di-esempio)
13. [Riferimenti](#13-riferimenti)

---

## 1. Il problema: TOP-TW

Il **Team Orienteering Problem with Time Windows** è una generalizzazione del Travelling Salesman Problem in cui:

- Esiste un insieme di **Punti di Interesse (PoI)** con uno score di attrattività, una durata di visita stimata e una finestra temporale di apertura `[open, close]`.
- Non tutti i PoI possono essere visitati entro il **budget di tempo** giornaliero.
- L'obiettivo è selezionare e ordinare un sottoinsieme di PoI in modo da **massimizzare lo score totale** rispettando i vincoli temporali.

In questo progetto il problema viene esteso con tre obiettivi competitivi simultanei:

| Obiettivo | Direzione | Significato |
|-----------|-----------|-------------|
| `total_score` | Massimizza | Interesse culturale/gastronomico del tour |
| `total_distance` | Minimizza | Fatica e costi di spostamento |
| `time_penalty` | Minimizza | Sforamento del budget giornaliero |

La natura multi-obiettivo rende la ricerca di un'unica soluzione ottima impossibile: esistono molteplici soluzioni di compromesso (fronte di Pareto) tra le quali il turista sceglie in base alle proprie preferenze.

---

## 2. Fondamenti teorici: NSGA-II

L'algoritmo NSGA-II (Deb, Pratap, Agarwal, Meyarivan — *IEEE Transactions on Evolutionary Computation*, Vol. 6, No. 2, 2002) risolve i tre limiti principali del suo predecessore NSGA:

1. **Complessità computazionale** — Il sorting non-dominato originale era O(MN³). NSGA-II lo riduce a **O(MN²)** con un algoritmo di conteggio della dominanza.
2. **Mancanza di elitismo** — NSGA-II combina popolazione corrente e figli in un pool di 2N individui e seleziona i migliori N: le soluzioni eccellenti non vengono mai perse.
3. **Parametro di sharing** — Sostituito dalla **crowding distance**, una metrica parameter-free che misura la densità delle soluzioni nell'intorno di un individuo nello spazio degli obiettivi.

### 2.1 Fast Non-Dominated Sorting

Per ogni individuo `i` si calcolano:
- `dom_count[i]`: numero di individui che dominano `i`
- `dom_set[i]`: insieme degli individui dominati da `i`

Un individuo `A` **domina** `B` se è migliore o uguale su tutti gli obiettivi e strettamente migliore su almeno uno. Gli individui con `dom_count = 0` formano il **Fronte 1** (Pareto front). Rimuovendo il fronte 1 e ripetendo si ottengono i fronti successivi. La complessità totale è **O(MN²)**.

### 2.2 Crowding Distance

Per ogni fronte, la crowding distance di un individuo `i` è la somma delle distanze normalizzate tra i vicini immediati lungo ciascun obiettivo:

```
CD(i) = Σ_m  |f_m(i+1) - f_m(i-1)| / (f_m_max - f_m_min)
```

Gli individui agli estremi del fronte ricevono distanza infinita. La crowding distance alta indica un individuo in una regione poco affollata — preferito nella selezione per mantenere diversità.

### 2.3 Crowded-Comparison Operator

Nella selezione a torneo, un individuo `A` è preferito a `B` se:
- `rank(A) < rank(B)` (rango Pareto migliore), oppure
- `rank(A) == rank(B)` e `crowd(A) > crowd(B)` (regione meno affollata)

### 2.4 Ciclo evolutivo principale

```
Inizializzazione popolazione P (greedy seeding + casuale riparato)
        ↓
Valutazione fitness (3 obiettivi)
        ↓
Fast non-dominated sort → assegna rank e crowding distance
        ↓
┌─────────────────────────────────────────────────────────┐
│  Selezione torneo (crowded-comparison)                  │
│  Crossover (OX | PoI-aware, prob cx_prob)               │
│  Mutazione (swap | insert | reverse | add-remove)       │
│  Riparazione (categoria → TW → budget → cap → pasto)   │
│  → popolazione figli Q (size N)                         │
└─────────────────────────────────────────────────────────┘
        ↓
Unione R = P ∪ Q  (size 2N)
        ↓
Non-dominated sort su R → seleziona migliori N → nuova P
        ↓
Criterio di stop? (max generazioni | stagnazione)
        ↓ NO                              ↓ SÌ
     (loop)                     Restituisce Fronte 1
```

### 2.5 Adattamenti al TOP-TW

L'NSGA-II standard opera su cromosomi di lunghezza fissa con variabili continue. Per il TOP-TW sono stati introdotti:

- **Cromosomi a lunghezza variabile** — lista ordinata di PoI (sottoinsieme del pool, non permutazione completa)
- **Repair-before-evaluation** — ogni individuo viene riparato dopo ogni operatore genetico, prima della valutazione fitness
- **Gene Jolly** (via `add_remove_mutation`) — placeholder che viene sostituito con il PoI più conveniente disponibile
- **Penalità dinamica sulle attese** — scoraggia tour con buchi temporali lunghi

---

## 3. Struttura del progetto

```
tour_generator_ga/
├── core/
│   ├── models.py       # PoI, Individual, TourSchedule, FitnessScore
│   ├── distance.py     # DistanceMatrix (Haversine × 1.3)
│   ├── fitness.py      # FitnessEvaluator multi-obiettivo
│   └── profile.py      # TouristProfile + modello trasporto realistico
├── ga/
│   ├── operators.py    # crossover (OX, PoI-aware), mutation, selection
│   ├── repair.py       # RepairEngine: 7 step di riparazione
│   └── seeding.py      # GreedySeeder: costruzione iniziale α-greedy
├── data/               # (placeholder per loader OSM/CSV)
├── solver.py           # NSGA2Solver: ciclo evolutivo completo
├── demo_rome.py        # Demo con dataset Roma e confronto profili
└── README.md
```

---

## 4. Modello dati

### `PoI`

```python
@dataclass
class PoI:
    id:             str
    name:           str
    lat, lon:       float
    score:          float           # attrattività [0, 1]
    visit_duration: int             # minuti di visita
    time_window:    TimeWindow      # (open, close) in minuti dalla mezzanotte
    category:       PoICategory     # MUSEUM | MONUMENT | RESTAURANT | BAR | GELATERIA | PARK | VIEWPOINT
    tags:           list[str]       # es. ["arte", "antico", "fotogenico"]
```

### `Individual` (cromosoma)

```python
class Individual:
    genes:    list[PoI]       # sequenza ordinata del tour
    fitness:  FitnessScore    # calcolata dopo evaluate()
    _schedule: TourSchedule   # cache dello schedule decodificato
```

La rappresentazione come lista ordinata permette di codificare naturalmente sia l'insieme dei PoI visitati sia l'ordine di visita, senza vincoli di lunghezza fissa.

### `TourSchedule`

Output della decodifica del cromosoma: per ogni PoI vengono calcolati orario di arrivo, attesa all'apertura e orario di partenza. Il campo `total_wait` traccia i minuti di attesa cumulati e contribuisce alla penalità fitness.

### `PoICategory`

```
MUSEUM | MONUMENT | RESTAURANT | BAR | GELATERIA | PARK | VIEWPOINT
```

La distinzione tra `RESTAURANT` (pasto formale) e `BAR`/`GELATERIA` (sosta breve) permette al `RepairEngine` di applicare vincoli differenziati per tipo di sosta alimentare.

---

## 5. Profilo turista (TouristProfile)

Il `TouristProfile` è l'oggetto centrale che attraversa l'intero pipeline e determina il comportamento di ogni componente.

```python
@dataclass
class TouristProfile:
    transport_mode:      TransportMode     # WALK | CAR | TRANSIT | MIXED
    mobility:            MobilityLevel     # NORMAL | LIMITED
    allowed_categories:  list[str]         # categorie PoI ammesse
    want_lunch:          bool              # richiede sosta pranzo
    want_dinner:         bool              # richiede sosta cena
    lunch_time:          int               # orario preferito pranzo (minuti)
    dinner_time:         int               # orario preferito cena (minuti)
    meal_window:         int               # flessibilità ±minuti attorno all'orario
    max_bar_stops:       int = 1           # max soste bar nel tour
    max_gelateria_stops: int = 1           # max soste gelateria nel tour
    tag_weights:         dict[str, float]  # boost per tag di interesse
    max_entry_fee:       float | None      # budget biglietti per PoI
    group_size:          int = 1           # persone (influenza durata visite)
```

**Profili predefiniti disponibili:**

| Factory | Modalità | Categorie | Pasti |
|---------|----------|-----------|-------|
| `profile_cultural_walker()` | WALK | museum, monument, viewpoint, restaurant | pranzo |
| `profile_foodie_transit()` | TRANSIT | restaurant, bar, gelateria, monument, viewpoint | pranzo + cena |
| `profile_family_mixed()` | MIXED / LIMITED | monument, park, viewpoint, restaurant | pranzo |
| `profile_art_lover_car()` | CAR | museum, monument | pranzo |

**Effetti sul pipeline:**

- `DistanceMatrix` — usa `profile.travel_time_min(km)` per i tempi di percorrenza
- `GreedySeeder` — filtra `allowed_pois` per categoria; usa `effective_score()` (con boost tag)
- `RepairEngine` — filtra categorie, applica cap ristoranti/snack, garantisce slot pasto
- `FitnessEvaluator` — calcola score con boost da `tag_weights`; penalizza pasti mancanti

---

## 6. Funzione fitness multi-obiettivo

La fitness è calcolata in `FitnessEvaluator.evaluate()` e comprende tre componenti separate per NSGA-II e uno scalare aggregato per confronti rapidi (tournament selection):

### Score effettivo con boost

```python
effective_score(poi) = min(poi.score × Π(1 + tag_weight - 1), 1.5)
```

I `tag_weights` del profilo amplificano i PoI tematicamente rilevanti (es. `arte × 1.4` per un turista culturale). Il cap a 1.5 evita distorsioni eccessive.

### Scalare aggregato

```
scalar = w_score × norm(total_score)
       - w_dist  × norm(total_distance)
       - time_over_penalty     # solo sullo sforamento, non sull'uso del budget
       - wait_penalty          # penalizza attese cumulate > 5 min
       - meal_penalty          # penalizza slot pasto non coperti
```

La scelta deliberata di **non penalizzare l'uso del budget** (solo lo sforamento) evita che il GA produca tour brevissimi per minimizzare il tempo. Un tour di 10 ore che rispetta il budget di 11 ore non è peggio di uno da 3 ore.

### I tre obiettivi Pareto

| Obiettivo | Misura | Direzione |
|-----------|--------|-----------|
| `total_score` | score effettivo cumulato | Massimizza |
| `total_distance` | km percorsi (Haversine × 1.3) | Minimizza |
| `total_time` | minuti totali del tour | Minimizza |

La dominanza Pareto è implementata in `FitnessScore.dominates()`:

```python
def dominates(self, other) -> bool:
    better_or_equal = (score ≥ and dist ≤ and time ≤)
    strictly_better = (score > or dist < or time <)
    return better_or_equal and strictly_better
```

---

## 7. Operatori genetici

### Selezione

**Tournament selection** con `k=3` candidati casuali. Con `use_pareto=True` (default) preferisce rango Pareto basso e crowding distance alta, implementando l'operatore crowded-comparison di NSGA-II.

### Crossover

**Order Crossover (OX) adattato** — Preserva l'ordine relativo dei PoI condivisi tra i genitori senza duplicati. Adattato al TOP-TW per gestire cromosomi a lunghezza variabile (sottoinsiemi, non permutazioni complete).

```
Genitore A: [Trevi, Navona, Pantheon, Pranzo, Colosseo]
Genitore B: [Colosseo, Pranzo, Borghese, Trevi, Castel]

Segmento da A (pos 1-2): [Navona, Pantheon]
Riempimento da B (escl. duplicati): [Colosseo, Pranzo, Trevi, Castel]
Figlio: [Colosseo, Navona, Pantheon, Pranzo, Trevi, Castel]
```

**PoI-aware Crossover** — Scambia interi blocchi per categoria (es. tutti i musei da A, tutti i ristoranti da B). Preserva la "nicchia tematica" del genitore e mantiene la coerenza del profilo turista.

### Mutazione

Quattro operatori applicati con probabilità adattiva:

| Operatore | Effetto | Caso d'uso |
|-----------|---------|------------|
| `swap_mutation` | Scambia 2 PoI nella sequenza | Esplorazione locale |
| `insert_mutation` | Rimuove e reinserisce un PoI | Fix ordinamento temporale |
| `reverse_segment_mutation` | Inverte un sottosegmento | Elimina crossing geografici |
| `add_remove_mutation` | Aggiunge/rimuove un PoI dal pool ammesso | Modifica lunghezza tour |

La `add_remove_mutation` opera **solo sul pool filtrato** per le categorie del profilo — non può mai inserire un ristorante nel tour di un turista che li ha esclusi.

---

## 8. Riparazione genetica (Repair Engine)

Ogni individuo viene riparato dopo ogni operatore genetico, prima della valutazione fitness. La pipeline di riparazione ha **7 step in sequenza**:

```
1. _filter_allowed_categories   → rimuove PoI di categorie non ammesse
2. _sort_by_earliest_deadline   → riordina per orario di apertura (EDF)
3. repair_time_windows          → rimuove PoI con TW violata o attesa > max_wait_min
4. repair_budget                → rimuove PoI a minor score/durata finché budget rispettato
5. _cap_restaurants             → limita ristoranti formali al numero di slot pasto
6. _cap_snacks                  → limita bar e gelaterie ai massimi del profilo
7. _ensure_meal_slots           → garantisce un ristorante per ogni slot pasto richiesto
```

### Step 3: max_wait_min

Il parametro `max_wait_min` (default 30) definisce la massima attesa tollerata per qualsiasi PoI. Un ristorante che apre alle 12:00 e a cui si arriva alle 9:30 viene **rimosso** — non ha senso aspettare 2 ore e mezza. L'eccezione è `_ensure_meal_slots`, che tolera fino a 45 minuti di attesa per garantire la sosta pranzo/cena.

### Step 7: strategia di inserimento/sostituzione

Quando deve garantire un pasto, `_ensure_meal_slots` prova due strategie:

1. **Inserimento diretto** — aggiunge il ristorante nella posizione temporalmente corretta senza sforare il budget.
2. **Sostituzione** — se non c'è spazio, rimuove il PoI con il peggior rapporto score/durata e lo sostituisce con il ristorante. Questo evita che tour a budget pieno perdano la garanzia del pasto.

---

## 9. Greedy Seeding

La popolazione iniziale è costruita con una strategia mista 20/20/60:

| Quota | Strategia | Scopo |
|-------|-----------|-------|
| 20% | Greedy puro (`alpha=0`) | Massima qualità iniziale, convergenza rapida |
| 20% | α-greedy perturbato (`alpha` da 0.15 a 0.50) | Varietà controllata vicino all'ottimo |
| 60% | Casuale riparato | Massima diversità genetica |

### Criterio greedy: ratio score/overhead

```python
ratio = effective_score(poi) / (travel_min + wait_min + visit_duration)
```

Dove `effective_score` include già i boost da `tag_weights` del profilo. Il criterio seleziona il PoI che massimizza il valore per minuto di tempo investito (spostamento + attesa + visita).

### Restricted Candidate List (GRASP-like)

Con `alpha > 0`, invece di scegliere sempre il candidato migliore, si sceglie **casualmente tra il top 20%** per ratio. Questo produce soluzioni diverse ma ancora di buona qualità, avvicinandosi alla strategia GRASP (Greedy Randomized Adaptive Search Procedure).

Tutti gli individui iniziali (greedy inclusi) passano per `repair.repair()` per garantire la coerenza con i vincoli del profilo fin dalla prima generazione.

---

## 10. Modello di trasporto realistico

Il calcolo dei tempi di percorrenza in `profile.travel_time_min(km)` usa un modello che riflette la realtà urbana:

### WALK

```
t = km / v_walk
```
- `v_walk_normal = 4.5 km/h`, `v_walk_limited = 3.0 km/h`

### TRANSIT (bus + metro)

```
if km < 0.40:          # soglia: prendere il mezzo non conviene
    t = km / v_walk
else:
    t = km / 20.0 + 10 min   # 10 min overhead (attesa + fermata)
```

L'overhead fisso di 10 minuti per tratta modella la realtà di Roma: frequenza media bus 8-12 minuti, metro 4-5 minuti, più il cammino alle fermate.

### CAR / TAXI

```
t = km / 25.0 + 5 min   # 5 min overhead parcheggio
```

### MIXED

Per distanze sotto `600 m` si usa `v_walk`; oltre si usa la velocità transit con overhead.

---

## 11. Configurazione e avvio

### Requisiti

```
Python ≥ 3.10   (per syntax X | None nei type hint)
Nessuna dipendenza esterna (stdlib only)
```

### SolverConfig

```python
from tour_ga.solver import SolverConfig

config = SolverConfig(
    pop_size         = 80,      # dimensione popolazione
    max_generations  = 300,     # generazioni massime
    cx_prob          = 0.85,    # probabilità crossover
    mut_prob         = 0.20,    # probabilità mutazione
    tournament_k     = 3,       # candidati per torneo
    stagnation_limit = 50,      # early stop per stagnazione
    start_time       = 540,     # 09:00 (minuti dalla mezzanotte)
    budget           = 480,     # 8 ore di tour
    start_lat        = 41.896,  # coordinate punto di partenza
    start_lon        = 12.484,
    max_wait_min     = 30,      # attesa massima tollerata per PoI
    w_score          = 0.50,    # peso obiettivo score
    w_dist           = 0.20,    # peso obiettivo distanza
    w_time           = 0.30,    # peso penalità tempo
)
```

### Avvio base

```python
from tour_ga.core.models import PoI, PoICategory, TimeWindow
from tour_ga.core.distance import DistanceMatrix
from tour_ga.core.profile import profile_cultural_walker
from tour_ga.solver import NSGA2Solver, SolverConfig

pois = [...]   # lista di PoI
profile = profile_cultural_walker()

dm = DistanceMatrix(pois, profile=profile)
dm.build()     # calcola matrice distanze (una volta sola)

solver = NSGA2Solver(pois, dm, config, profile=profile)
pareto_front = solver.solve()  # restituisce lista di Individual

# Selezione della soluzione consigliata dal fronte
best = max(pareto_front, key=lambda x: x.fitness.scalar)
schedule = solver.evaluator.decode(best)
print(schedule.summary())
```

### Profilo custom

```python
from tour_ga.core.profile import TouristProfile, TransportMode

profile = TouristProfile(
    transport_mode     = TransportMode.TRANSIT,
    allowed_categories = ["monument", "viewpoint", "bar"],
    want_lunch         = False,
    want_dinner        = False,
    max_bar_stops      = 2,
    tag_weights        = {"fotogenico": 1.5, "panorama": 1.3},
)
```

### Callback di monitoraggio

```python
def progress(gen, pareto_front, stats):
    print(f"Gen {gen}: pareto={stats['pareto_size']}, "
          f"best={stats['best_scalar']:.4f}, "
          f"feasible={stats['feasible_pct']:.0f}%")

front = solver.solve(callback=progress)
```

### Demo completa

```bash
python tour_ga/demo_rome.py
```

---

## 12. Risultati di esempio

Configurazione: `budget=660 min` (11h), `start_time=570` (09:30), `pop_size=60`, `max_generations=200`, Roma.

### Profilo gastronomico con mezzi (TRANSIT, pranzo + cena)

```
09:42–10:12  Fontana di Trevi
10:24–10:54  Piazza di Spagna
11:08–11:53  Piazza Navona
11:56–12:16  Sant'Eustachio il Caffè         ← bar pomeridiano
12:27–13:27  Osteria del Rione               ← pranzo garantito
13:41–15:11  Castel Sant'Angelo
15:24–16:24  Pantheon
16:37–18:07  Foro Romano
18:21–18:41  Fatamorgana                     ← gelateria pomeridiana
19:00–20:20  Ristorante San Pietro           ← cena garantita (attesa 4 min)
Totale: 650 min, 10.9 km, attese 4 min
Composizione: 1×bar, 1×gelateria, 4×monument, 2×restaurant, 2×viewpoint
```

### Profilo culturale a piedi (WALK, solo pranzo)

```
09:39–10:09  Fontana di Trevi
10:18–10:48  Piazza di Spagna
11:06–11:51  Piazza Navona
12:00–13:00  Osteria del Rione               ← pranzo (attesa 3 min)
13:05–14:05  Pantheon
14:21–15:51  Foro Romano
16:01–18:01  Colosseo
18:32–20:02  Trastevere
Totale: 632 min, 8.1 km, attese 3 min
```

### Profilo custom: solo viste, nessun pasto

```
09:39–10:09  Fontana di Trevi
10:18–10:48  Piazza di Spagna
11:06–11:51  Piazza Navona
11:54–12:14  Sant'Eustachio il Caffè
12:16–13:16  Pantheon
13:33–15:03  Castel Sant'Angelo
15:14–15:34  Fatamorgana
Totale: 364 min, 5.5 km
```

---

## 13. Riferimenti

### Articolo scientifico principale

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). **A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II**. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
[https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf](https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf)

Il paper introduce tre contributi fondamentali qui implementati: il fast non-dominated sorting O(MN²) tramite conteggio della dominanza (implementato in `solver.py::_fast_non_dominated_sort`), la crowding distance come meccanismo di diversità parameter-free (implementato in `_assign_crowding_distance`), e il crowded-comparison operator per selezione a torneo (in `operators.py::tournament_select`).

### Riferimento divulgativo

Non-Dominated Sorting Genetic Algorithm 2 (NSGA-II) — GeeksforGeeks.
[https://www.geeksforgeeks.org/deep-learning/non-dominated-sorting-genetic-algorithm-2-nsga-ii/](https://www.geeksforgeeks.org/deep-learning/non-dominated-sorting-genetic-algorithm-2-nsga-ii/)

### Problema di riferimento

Vansteenwegen, P., Souffriau, W., & Van Oudheusden, D. (2011). **The Orienteering Problem: A Survey**. *European Journal of Operational Research*, 209(1), 1–10.

Chao, I. M., Golden, B. L., & Wasil, E. A. (1996). **The Team Orienteering Problem**. *European Journal of Operational Research*, 88(3), 464–474.

### Calcolo distanze

Formula di Haversine per la distanza geodetica tra due coordinate GPS, con fattore correttivo 1.3 per approssimare il percorso urbano reale rispetto alla linea d'aria.