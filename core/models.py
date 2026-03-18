"""
core/models.py — Strutture dati fondamentali per il TOP-TW turistico.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


class PoICategory(Enum):
    MUSEUM    = "museum"
    MONUMENT  = "monument"
    RESTAURANT = "restaurant"
    PARK      = "park"
    VIEWPOINT = "viewpoint"


@dataclass
class TimeWindow:
    open:  int   # minuti dalla mezzanotte (es. 540 = 09:00)
    close: int   # minuti dalla mezzanotte (es. 1080 = 18:00)

    def __repr__(self) -> str:
        return f"{self.open//60:02d}:{self.open%60:02d}–{self.close//60:02d}:{self.close%60:02d}"


@dataclass
class PoI:
    id:             str
    name:           str
    lat:            float
    lon:            float
    score:          float          # interesse normalizzato [0, 1]
    visit_duration: int            # minuti di visita stimati
    time_window:    TimeWindow
    category:       PoICategory
    tags:           list[str]      = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, PoI) and self.id == other.id

    def __repr__(self):
        return f"PoI({self.name!r}, score={self.score:.2f}, {self.time_window})"


@dataclass
class FitnessScore:
    total_score:    float = 0.0   # somma score PoI visitati
    total_distance: float = 0.0   # km totali percorsi
    total_time:     int   = 0     # minuti totali (spostamenti + visite)
    is_feasible:    bool  = False  # rispetta TW e budget?
    scalar:         float = 0.0   # valore aggregato per confronti rapidi
    rank:           int   = 0     # rango Pareto (NSGA-II)
    crowd:          float = 0.0   # crowding distance (NSGA-II)

    def dominates(self, other: FitnessScore) -> bool:
        """
        self domina other se è ≥ su tutti gli obiettivi e > su almeno uno.
        Obiettivi: massimizza score, minimizza distance, minimizza time.
        """
        better_or_equal = (
            self.total_score    >= other.total_score and
            self.total_distance <= other.total_distance and
            self.total_time     <= other.total_time
        )
        strictly_better = (
            self.total_score    > other.total_score or
            self.total_distance < other.total_distance or
            self.total_time     < other.total_time
        )
        return better_or_equal and strictly_better


@dataclass
class ScheduledStop:
    poi:         PoI
    arrival:     int   # minuti dalla mezzanotte
    departure:   int   # minuti dalla mezzanotte
    wait:        int   # minuti di attesa prima dell'apertura


@dataclass
class TourSchedule:
    stops:         list[ScheduledStop] = field(default_factory=list)
    total_time:    int   = 0
    total_distance: float = 0.0
    is_feasible:   bool  = False

    def summary(self) -> str:
        lines = []
        for s in self.stops:
            a = f"{s.arrival//60:02d}:{s.arrival%60:02d}"
            d = f"{s.departure//60:02d}:{s.departure%60:02d}"
            w = f" (attesa {s.wait} min)" if s.wait > 0 else ""
            lines.append(f"  {a}–{d}  {s.poi.name}{w}")
        lines.append(f"  Totale: {self.total_time} min, {self.total_distance:.1f} km")
        return "\n".join(lines)


class Individual:
    """
    Cromosoma = lista ordinata di PoI che compongono il tour.
    Il gene jolly (WildcardGene) è un placeholder che viene
    materializzato al momento della decodifica.
    """

    def __init__(self, genes: list[PoI]):
        self.genes:    list[PoI]        = genes
        self.fitness:  FitnessScore     = FitnessScore()
        self._schedule: Optional[TourSchedule] = None  # cache

    def clone(self) -> Individual:
        return Individual(genes=list(self.genes))

    def invalidate_cache(self):
        self._schedule = None
        self.fitness   = FitnessScore()

    def __len__(self):
        return len(self.genes)

    def __repr__(self):
        names = [p.name for p in self.genes]
        return f"Individual([{', '.join(names)}], scalar={self.fitness.scalar:.3f})"