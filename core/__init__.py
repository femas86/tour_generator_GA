"""
tour_generator_GA.core — Strutture dati e logica di valutazione del problema TOP-TW.

Esporta le classi principali per comodità di importazione:
    from tour_generator_GA.core import PoI, Individual, FitnessEvaluator, DistanceMatrix
"""
from .models import (
    PoI,
    PoICategory,
    TimeWindow,
    FitnessScore,
    Individual,
    TourSchedule,
    ScheduledStop,
)
from .distance import DistanceMatrix, haversine_km
from .fitness import FitnessEvaluator

__all__ = [
    "PoI",
    "PoICategory",
    "TimeWindow",
    "FitnessScore",
    "Individual",
    "TourSchedule",
    "ScheduledStop",
    "DistanceMatrix",
    "haversine_km",
    "FitnessEvaluator",
]