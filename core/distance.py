"""
core/distance.py — Matrice delle distanze e tempi di percorrenza.
Supporta Haversine (offline) e profilo turista per la velocità.
"""
from __future__ import annotations
import math
from typing import Union, Optional, TYPE_CHECKING
from .models import PoI
from config import ROUTE_DETOUR_FACTOR

if TYPE_CHECKING:
    from .profile import TouristProfile


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
    Precalcola tutte le distanze tra i PoI (km).
    I TEMPI vengono calcolati on-the-fly tramite il TouristProfile,
    così un cambio di modalità non richiede di ricostruire la matrice.
    """

    def __init__(self, pois: list[PoI], profile: Optional["TouristProfile"] = None):
        self.pois    = pois
        self.profile = profile
        self.idx     = {poi.id: i for i, poi in enumerate(pois)}
        n            = len(pois)
        self._dist   = [[0.0] * n for _ in range(n)]  # km (invariante)

    def build(self):
        """Popola la matrice delle distanze. Chiama una volta sola."""
        for i, a in enumerate(self.pois):
            for j, b in enumerate(self.pois):
                if i == j:
                    continue
                km = haversine_km(a.lat, a.lon, b.lat, b.lon) * ROUTE_DETOUR_FACTOR
                self._dist[i][j] = km

    def dist(self, a: Union[PoI, str], b: Union[PoI, str]) -> float:
        """Distanza in km tra due PoI."""
        ia = self.idx[a.id if isinstance(a, PoI) else a]
        ib = self.idx[b.id if isinstance(b, PoI) else b]
        return self._dist[ia][ib]

    def time(self, a: Union[PoI, str], b: Union[PoI, str]) -> int:
        """Tempo di percorrenza in minuti, rispettando la modalità del profilo."""
        km = self.dist(a, b)
        return self._km_to_min(km)

    def time_from_coord(self, lat: float, lon: float, poi: PoI) -> int:
        """Tempo in minuti da coordinate arbitrarie (es. hotel) a un PoI."""
        km = haversine_km(lat, lon, poi.lat, poi.lon) * ROUTE_DETOUR_FACTOR
        return self._km_to_min(km)

    def _km_to_min(self, km: float) -> int:
        if self.profile is not None:
            return self.profile.travel_time_min(km)
        # Fallback sicuro: a piedi 4.5 km/h
        return max(1, int((km / 4.5) * 60))