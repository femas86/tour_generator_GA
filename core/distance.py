"""
core/distance.py — Matrice delle distanze e tempi di percorrenza.
Supporta Haversine (offline) e OSRM (online, opzionale).
"""
from __future__ import annotations
import math
from functools import lru_cache
from typing import Union
from .models import PoI


# Velocità medie di spostamento a piedi / mezzi pubblici (km/h)
WALK_SPEED_KMH = 4.5
TRANSIT_SPEED_KMH = 20.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distanza geodetica tra due coordinate in chilometri."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))


class DistanceMatrix:
    """
    Precalcola tutte le distanze e i tempi di percorrenza tra i PoI.
    Usare build() prima di avviare il GA per evitare calcoli ripetuti.
    """

    def __init__(self, pois: list[PoI], mode: str = "walk"):
        self.pois    = pois
        self.mode    = mode
        self.idx     = {poi.id: i for i, poi in enumerate(pois)}
        n            = len(pois)
        self._dist   = [[0.0] * n for _ in range(n)]  # km
        self._time   = [[0]   * n for _ in range(n)]  # minuti

    def build(self):
        """Popola la matrice con Haversine. Chiama una volta sola."""
        speed = WALK_SPEED_KMH if self.mode == "walk" else TRANSIT_SPEED_KMH
        for i, a in enumerate(self.pois):
            for j, b in enumerate(self.pois):
                if i == j:
                    continue
                km = haversine_km(a.lat, a.lon, b.lat, b.lon)
                # Fattore 1.3 per percorso reale vs linea d'aria
                km *= 1.3
                self._dist[i][j] = km
                self._time[i][j] = int((km / speed) * 60)  # minuti

    def dist(self, a: Union[PoI, str], b: Union[PoI, str]) -> float:
        """Distanza in km tra due PoI."""
        ia = self.idx[a.id if isinstance(a, PoI) else a]
        ib = self.idx[b.id if isinstance(b, PoI) else b]
        return self._dist[ia][ib]

    def time(self, a: Union[PoI, str], b: Union[PoI, str]) -> int:
        """Tempo di percorrenza in minuti tra due PoI."""
        ia = self.idx[a.id if isinstance(a, PoI) else a]
        ib = self.idx[b.id if isinstance(b, PoI) else b]
        return self._time[ia][ib]

    def time_from_start(self, start_lat: float, start_lon: float, poi: PoI) -> int:
        """Tempo in minuti dalla posizione di partenza (hotel/stazione) a un PoI."""
        speed = WALK_SPEED_KMH if self.mode == "walk" else TRANSIT_SPEED_KMH
        km = haversine_km(start_lat, start_lon, poi.lat, poi.lon) * 1.3
        return int((km / speed) * 60)