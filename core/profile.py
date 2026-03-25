"""
core/profile.py — Profilo del turista con tutte le preferenze personali.
È l'oggetto centrale che attraversa TUTTO il pipeline GA:
  DistanceMatrix → velocità di spostamento per modalità
  GreedySeeder   → whitelist categorie + slot pasto garantito
  RepairEngine   → blacklist categorie + rimozione violazioni profilo
  FitnessEvaluator → boost/malus score per tag di interesse
  GeneticOperators → add/remove usa solo PoI ammessi dal profilo
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from config import (MIXED_THRESHOLD_M, TRANSIT_WALK_THRESHOLD_KM, SCORE_BOOST_CAP,
                    TRANSIT_OVERHEAD_MIN, CAR_OVERHEAD_MIN, SPEED_KMH,
                    MIXED_LONG_SPEED_KMH, WANT_LUNCH, WANT_DINNER, LUNCH_TIME,
                    DINNER_TIME, MEAL_WINDOW, MAX_BAR_STOPS, MAX_GELATERIA_STOPS)


class TransportMode(Enum):
    WALK    = "walk"      # tutto a piedi (centro storico)
    CAR     = "car"       # auto / taxi
    TRANSIT = "transit"   # bus + metro
    MIXED   = "mixed"     # a piedi <MIXED_THRESHOLD_M, transit/auto oltre


class MobilityLevel(Enum):
    NORMAL  = "normal"    # nessuna limitazione
    LIMITED = "limited"   # difficoltà con scale, distanze lunghe → penalizza PoI lontani


# Soglia in metri sotto la quale si va a piedi anche in modalità MIXED
#MIXED_THRESHOLD_M = 600

# Soglia al di sotto della quale, anche in modalità TRANSIT, si cammina:
# prendere un mezzo per 300m è spesso più lento che andare a piedi.
#TRANSIT_WALK_THRESHOLD_KM = 0.40

# Overhead fisso per ogni tratta in mezzo pubblico (minuti):
# comprende cammino alla fermata + attesa + cammino dalla fermata.
# Roma: metro ha frequenza ~5 min, bus ~8-12 min → media ~10 min overhead.
#TRANSIT_OVERHEAD_MIN = 10

# Overhead per auto/taxi: parcheggio + spostamento a piedi dal parcheggio.
#CAR_OVERHEAD_MIN = 5

# Velocità medie di percorrenza (escluso overhead)
# SPEED_TABLE: dict[tuple[TransportMode, MobilityLevel], float] = {
#     (TransportMode.WALK,    MobilityLevel.NORMAL):  4.5,
#     (TransportMode.WALK,    MobilityLevel.LIMITED): 3.0,
#     (TransportMode.CAR,     MobilityLevel.NORMAL):  25.0,
#     (TransportMode.CAR,     MobilityLevel.LIMITED): 25.0,
#     (TransportMode.TRANSIT, MobilityLevel.NORMAL):  20.0,  # velocità effettiva metro/bus Roma
#     (TransportMode.TRANSIT, MobilityLevel.LIMITED): 16.0,
#     (TransportMode.MIXED,   MobilityLevel.NORMAL):  4.5,   # usata per segmenti brevi
#     (TransportMode.MIXED,   MobilityLevel.LIMITED): 3.0,
# }

# MIXED_LONG_SPEED: dict[MobilityLevel, float] = {
#     MobilityLevel.NORMAL:  20.0,
#     MobilityLevel.LIMITED: 16.0,
# }


@dataclass
class TouristProfile:
    """
    Preferenze complete del turista.
    Tutti i campi hanno un default sensato per un turista generico.
    """

    # --- Trasporto ---
    transport_mode: TransportMode = TransportMode.WALK
    mobility:       MobilityLevel = MobilityLevel.NORMAL

    # --- Categorie ammesse ---
    # Se una categoria non è in questa lista, i suoi PoI vengono
    # completamente ignorati in seeding, repair e mutation.
    allowed_categories: list[str] = field(default_factory=lambda: [
        "museum", "monument", "restaurant", "park", "viewpoint"
    ])

    # --- Preferenze pasti ---
    want_lunch:  bool = WANT_LUNCH
    want_dinner: bool = WANT_DINNER
    lunch_time:  int  = LUNCH_TIME
    dinner_time: int  = DINNER_TIME
    meal_window: int  = MEAL_WINDOW

    # --- Soste snack (bar, gelateria) ---
    # Numero massimo di soste snack per tipo nel tour.
    # None = nessun limite (utile per profili food-focused).
    max_bar_stops:       int = MAX_BAR_STOPS
    max_gelateria_stops: int = MAX_GELATERIA_STOPS

    # --- Interessi tematici (tag) ---
    # Ogni tag elencato riceve un boost moltiplicativo allo score del PoI.
    # Es. {"arte": 1.5, "antico": 1.3} → i musei d'arte valgono 50% di più.
    tag_weights: dict[str, float] = field(default_factory=dict)

    # --- Budget economico ---
    max_entry_fee: Optional[float] = None   # euro; None = nessun limite

    # --- Gruppo ---
    group_size: int = 1   # utile per entry fee totale e ritmo di visita

    def __post_init__(self):
        # Normalizza le categorie in minuscolo
        self.allowed_categories = [c.lower() for c in self.allowed_categories]

    def allows_category(self, category_value: str) -> bool:
        """Restituisce True se la categoria è ammessa dal profilo."""
        return category_value.lower() in self.allowed_categories

    def effective_score(self, poi) -> float:
        """
        Score del PoI moltiplicato per i boost dei tag di interesse.
        Un PoI senza tag corrispondenti mantiene lo score base.
        """
        boost = 1.0
        for tag in poi.tags:
            if tag in self.tag_weights:
                boost += self.tag_weights[tag] - 1.0  # additive boost
        return min(poi.score * boost, SCORE_BOOST_CAP)  
    
    def travel_speed_kmh(self, dist_km: float) -> float:
        """Velocità di percorrenza pura (senza overhead fisso)."""
        if self.transport_mode == TransportMode.MIXED:
            dist_m = dist_km * 1000
            if dist_m <= MIXED_THRESHOLD_M:
                return SPEED_KMH[(TransportMode.WALK.value, self.mobility.value)]
            else:
                return MIXED_LONG_SPEED_KMH[self.mobility]
        return SPEED_KMH.get((self.transport_mode.value, self.mobility.value))

    def travel_time_min(self, dist_km: float) -> int:
        """
        Tempo di percorrenza realistico in minuti.

        Modello per modalità:
          WALK    → dist / v_walk
          CAR     → dist / v_car  + CAR_OVERHEAD (parcheggio)
          TRANSIT → se dist < soglia: a piedi (prendere il mezzo non conviene)
                    altrimenti: dist / v_transit + TRANSIT_OVERHEAD (attesa + fermata)
          MIXED   → a piedi se dist < MIXED_THRESHOLD, altrimenti come TRANSIT
        """
        mode = self.transport_mode
        walk_speed = SPEED_KMH[(TransportMode.WALK.value, self.mobility.value)]

        if mode == TransportMode.WALK:
            return max(1, int((dist_km / walk_speed) * 60))

        if mode == TransportMode.CAR:
            speed = SPEED_KMH.get((TransportMode.CAR.value, self.mobility.value))
            return max(3, int((dist_km / speed) * 60) + CAR_OVERHEAD_MIN)

        if mode == TransportMode.TRANSIT:
            if dist_km < TRANSIT_WALK_THRESHOLD_KM:
                # Distanza troppo corta: a piedi è più veloce del mezzo
                return max(1, int((dist_km / walk_speed) * 60))
            speed = SPEED_KMH[(TransportMode.TRANSIT.value, self.mobility.value)]
            ride  = int((dist_km / speed) * 60)
            return ride + TRANSIT_OVERHEAD_MIN

        if mode == TransportMode.MIXED:
            dist_m = dist_km * 1000
            if dist_m <= MIXED_THRESHOLD_M:
                return max(1, int((dist_km / walk_speed) * 60))
            long_speed = MIXED_LONG_SPEED_KMH[self.mobility.value]
            ride = int((dist_km / long_speed) * 60)
            return ride + TRANSIT_OVERHEAD_MIN

        # Fallback
        return max(1, int((dist_km / walk_speed) * 60))

    def needs_meal_slot(self) -> list[tuple[int, int]]:
        """
        Restituisce la lista di finestre temporali in cui il profilo
        richiede un ristorante nel tour.
        Es. [(660, 780), (1080, 1200)] per pranzo+cena.
        """
        slots = []
        if self.want_lunch:
            slots.append((
                self.lunch_time - self.meal_window,
                self.lunch_time + self.meal_window
            ))
        if self.want_dinner:
            slots.append((
                self.dinner_time - self.meal_window,
                self.dinner_time + self.meal_window
            ))
        return slots

    def summary(self) -> str:
        lines = [
            f"Trasporto : {self.transport_mode.value}  |  Mobilità: {self.mobility.value}",
            f"Categorie : {', '.join(self.allowed_categories)}",
            f"Pranzo    : {'sì (' + str(self.lunch_time//60) + ':00)' if self.want_lunch else 'no'}  |  "
            f"Cena: {'sì (' + str(self.dinner_time//60) + ':00)' if self.want_dinner else 'no'}",
        ]
        if self.tag_weights:
            tw = ", ".join(f"{k}×{v}" for k, v in self.tag_weights.items())
            lines.append(f"Interessi : {tw}")
        if self.max_entry_fee is not None:
            lines.append(f"Budget biglietti: €{self.max_entry_fee:.0f} max a PoI")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory: profili predefiniti pronti all'uso
# ---------------------------------------------------------------------------
#TODO: rivedere la definizione di tutti i profili e i pesi, facendo attenzione alle allowed_categories e alle tag_weights per coerenza interna.
def profile_cultural_walker() -> TouristProfile:
    """Turista culturale a piedi, interessato ad arte e storia. Include una sosta pranzo."""
    return TouristProfile(
        transport_mode     = TransportMode.WALK,
        allowed_categories = ["museum", "monument", "viewpoint", "restaurant", "bar", "gelateria"],
        want_lunch         = True,
        want_dinner        = False,
        tag_weights        = {"arte": 1.4, "antico": 1.3, "rinascimento": 1.5, "unesco": 1.2},
        max_bar_stops      = 1,
        max_gelateria_stops= 1
    )


def profile_foodie_transit() -> TouristProfile:
    """Turista gastronomico con mezzi pubblici, ristoranti inclusi."""
    return TouristProfile(
        transport_mode     = TransportMode.TRANSIT,
        allowed_categories = ["restaurant", "monument", "bar", "gelateria", "viewpoint", "park"],
        want_lunch         = True,
        want_dinner        = True,
        tag_weights        = {"cucina_romana": 1.6, "offal": 0.5, "vivace": 1.2},
        max_bar_stops      = 2,
        max_gelateria_stops= 2
    )


def profile_family_mixed() -> TouristProfile:
    """Famiglia con bambini: percorso misto, evita musei pesanti."""
    return TouristProfile(
        transport_mode     = TransportMode.MIXED,
        mobility           = MobilityLevel.LIMITED,
        allowed_categories = ["monument", "park", "viewpoint", "restaurant", "bar", "gelateria"],
        want_lunch         = True,
        want_dinner        = False,
        group_size         = 4,
        tag_weights        = {"fotogenico": 1.3, "vivace": 1.2},
        max_entry_fee      = 15.0,
        max_bar_stops      = 1,
        max_gelateria_stops= 1
    )


def profile_art_lover_car() -> TouristProfile:
    """Appassionato d'arte con auto: vuole visitare musei anche lontani."""
    return TouristProfile(
        transport_mode     = TransportMode.CAR,
        allowed_categories = ["museum", "monument", "bar", "gelateria"],
        want_lunch         = True,
        want_dinner        = False,
        tag_weights        = {"arte": 1.6, "scultura": 1.5, "rinascimento": 1.4, "antico": 1.1},
        max_entry_fee      = 30.0,
        max_bar_stops      = 1,
        max_gelateria_stops= 1
        
    )