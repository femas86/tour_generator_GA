"""
app.py — FastAPI backend for Tour Generator.
Exposes API endpoints for generating tours.
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import json
from typing import List, Optional
from core.models import PoI, PoICategory, TimeWindow
from core.profile import (
    TouristProfile, MobilityLevel, TransportMode,
    profile_cultural_walker, profile_foodie_transit,
    profile_family_mixed, profile_art_lover_car
)
from core.distance import DistanceMatrix, haversine_km
from solver import NSGA2Solver, SolverConfig
import pandas as pd

app = FastAPI(title="Tour Generator API", description="API for generating optimized tours using genetic algorithms")

# Predefined profiles
PREDEFINED_PROFILES = {
    "cultural_walker": profile_cultural_walker(),
    "foodie_transit": profile_foodie_transit(),
    "family_mixed": profile_family_mixed(),
    "art_lover_car": profile_art_lover_car(),
}

class PoIModel(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    score: float
    visit_duration: int
    time_window_open: int
    time_window_close: int
    category: str
    tags: List[str] = []

class ProfileModel(BaseModel):
    transport_mode: TransportMode = TransportMode.WALK
    mobility: MobilityLevel = MobilityLevel.NORMAL
    allowed_categories: List[str] = ["museum", "monument", "restaurant", "park", "viewpoint"]
    want_lunch: bool = True
    want_dinner: bool = True
    lunch_time: int = 720
    dinner_time: int = 1140
    meal_window: int = 120
    max_bar_stops: int = 2
    max_gelateria_stops: int = 1
    tag_weights: dict = {}
    max_entry_fee: Optional[float] = None
    group_size: int = 1

@app.post("/generate_tour")
async def generate_tour(
    pois_file: Optional[UploadFile] = File(None),
    pois_json: Optional[str] = Form(None),
    profile_name: Optional[str] = Form(None),
    profile_json: Optional[str] = Form(None),
    budget: int = Form(480),
    start_time: int = Form(540),
    start_lat: float = Form(41.9028),
    start_lon: float = Form(12.4964),
):
    """
    Generate an optimized tour based on POIs and user profile.
    
    - pois_file: Upload CSV or JSON file with POIs
    - pois_json: JSON string with list of POIs
    - profile_name: Name of predefined profile
    - profile_json: JSON string with custom profile
    - budget: Time budget in minutes
    - start_time: Start time in minutes from midnight
    - start_lat/lon: Starting coordinates
    """
    # Load POIs
    pois = []
    if pois_file:
        content = await pois_file.read()
        if pois_file.filename.endswith('.csv'):
            df = pd.read_csv(pd.io.common.BytesIO(content))
            for _, row in df.iterrows():
                pois.append(PoI(
                    id=str(row['id']),
                    name=str(row['name']),
                    lat=float(row['lat']),
                    lon=float(row['lon']),
                    score=float(row['score']),
                    visit_duration=int(row['visit_duration']),
                    time_window=TimeWindow(int(row['time_window_open']), int(row['time_window_close'])),
                    category=PoICategory(row['category']),
                    tags=str(row.get('tags', '')).split(',') if pd.notna(row.get('tags')) else []
                ))
        elif pois_file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            for p in data:
                pois.append(PoI(
                    id=p['id'],
                    name=p['name'],
                    lat=p['lat'],
                    lon=p['lon'],
                    score=p['score'],
                    visit_duration=p['visit_duration'],
                    time_window=TimeWindow(p['time_window']['open'], p['time_window']['close']),
                    category=PoICategory(p['category']),
                    tags=p.get('tags', [])
                ))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for POIs. Use CSV or JSON.")
    elif pois_json:
        pois_data = json.loads(pois_json)
        for p in pois_data:
            pois.append(PoI(
                id=p['id'],
                name=p['name'],
                lat=p['lat'],
                lon=p['lon'],
                score=p['score'],
                visit_duration=p['visit_duration'],
                time_window=TimeWindow(p['time_window']['open'], p['time_window']['close']),
                category=PoICategory(p['category']),
                tags=p.get('tags', [])
            ))
    else:
        raise HTTPException(status_code=400, detail="POIs not provided. Upload a file or provide JSON.")

    # Load profile
    if profile_name:
        if profile_name in PREDEFINED_PROFILES:
            profile = PREDEFINED_PROFILES[profile_name]
        else:
            raise HTTPException(status_code=400, detail=f"Invalid profile name. Available: {list(PREDEFINED_PROFILES.keys())}")
    elif profile_json:
        profile_data = json.loads(profile_json)
        profile = TouristProfile(**profile_data)
    else:
        profile = TouristProfile()  # default

    # Create distance matrix
    dm = DistanceMatrix(pois, profile)

    # Config
    config = SolverConfig(budget=budget, start_time=start_time, start_lat=start_lat, start_lon=start_lon)

    # Solve
    def cb(gen, pareto, stats):
        if gen % 30 == 0 or gen == 1:
            print(f"  gen {gen:3d} | pareto={stats['pareto_size']:2d} | "
                  f"best={stats['best_scalar']:.4f} | feasible={stats['feasible_pct']:.0f}%")

    solver = NSGA2Solver(pois, dm, config, profile)
    population = solver.solve(callback=cb)

    feasible = [x for x in population if x.fitness.is_feasible] or population
    if not feasible:
        raise HTTPException(status_code=500, detail="No solutions found")

    # Get best tour (highest scalar fitness)
    best = max(feasible, key=lambda individual: individual.fitness.scalar)
    tour = solver.evaluator.decode(best)

    if tour is None:
        raise HTTPException(status_code=500, detail="Failed to generate schedule")

    # Return as dict
    stops_list = []
    for i, s in enumerate(tour.stops):
        if i == 0:
            dist = haversine_km(start_lat, start_lon, s.poi.lat, s.poi.lon)
        else:
            dist = haversine_km(tour.stops[i-1].poi.lat, tour.stops[i-1].poi.lon, s.poi.lat, s.poi.lon)
        time_min = profile.travel_time_min(dist)
        
        stop_dict = {
            "poi_id": s.poi.id,
            "poi_name": s.poi.name,
            "arrival": s.arrival,
            "departure": s.departure,
            "wait": s.wait,
            "travel_distance_km": round(dist, 2),
            "travel_time_min": time_min
        }
        stops_list.append(stop_dict)
    
    return {
        "total_score": best.fitness.total_score,
        "total_distance": tour.total_distance,
        "total_time": tour.total_time,
        "is_feasible": tour.is_feasible,
        "stops": stops_list
    }

@app.get("/profiles")
def get_profiles():
    """Get list of available predefined profiles."""
    return {"profiles": list(PREDEFINED_PROFILES.keys())}

@app.get("/profiles/{name}")
def get_profile(name: str):
    """Get details of a specific predefined profile."""
    if name in PREDEFINED_PROFILES:
        profile = PREDEFINED_PROFILES[name]
        return {
            "transport_mode": profile.transport_mode.value,
            "mobility": profile.mobility.value,
            "allowed_categories": profile.allowed_categories,
            "want_lunch": profile.want_lunch,
            "want_dinner": profile.want_dinner,
            "lunch_time": profile.lunch_time,
            "dinner_time": profile.dinner_time,
            "meal_window": profile.meal_window,
            "max_bar_stops": profile.max_bar_stops,
            "max_gelateria_stops": profile.max_gelateria_stops,
            "tag_weights": profile.tag_weights,
            "max_entry_fee": profile.max_entry_fee,
            "group_size": profile.group_size
        }
    else:
        raise HTTPException(status_code=404, detail="Profile not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)