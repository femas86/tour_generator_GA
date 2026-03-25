"""
streamlit_ui.py — Streamlit UI for Tour Generator.
Allows uploading POIs and selecting profile to generate tours.
"""
import streamlit as st
import requests
import json
import os
import pandas as pd
from typing import Optional

st.title("🗺️ Tour Generator")
st.markdown("Upload Points of Interest and select a user profile to generate an optimized tour using genetic algorithms.")

# Get available profiles
try:
    response = requests.get("http://localhost:8000/profiles")
    if response.status_code == 200:
        available_profiles = response.json()["profiles"]
    else:
        available_profiles = ["cultural_walker", "foodie_transit", "family_mixed", "art_lover_car"]
except:
    available_profiles = ["cultural_walker", "foodie_transit", "family_mixed", "art_lover_car"]

st.header("📍 Upload Points of Interest (POIs)")
pois_file = st.file_uploader(
    "Upload POIs as CSV or JSON file",
    type=['csv', 'json'],
    help="CSV columns: id,name,lat,lon,score,visit_duration,time_window_open,time_window_close,category,tags\nJSON: list of POI objects"
)

st.header("👤 Select User Profile")
profile_option = st.radio("Profile Type", ["Predefined", "Custom"])

profile_name: Optional[str] = None
profile_data: Optional[dict] = None

if profile_option == "Predefined":
    profile_name = st.selectbox("Choose a predefined profile", available_profiles)
    if profile_name:
        try:
            response = requests.get(f"http://localhost:8000/profiles/{profile_name}")
            if response.status_code == 200:
                profile_data = response.json()
                st.subheader(f"Profile Details: {profile_name}")
                st.json(profile_data)
        except:
            st.warning("Could not load profile details")
else:
    profile_file = st.file_uploader(
        "Upload custom profile JSON",
        type=['json'],
        help="JSON object with TouristProfile fields"
    )
    if profile_file:
        try:
            profile_data = json.load(profile_file)
            st.json(profile_data)
        except:
            st.error("Invalid JSON file")

st.header("⚙️ Tour Parameters")
col1, col2 = st.columns(2)
with col1:
    budget = st.number_input("Time Budget (minutes)", value=480, min_value=60, help="Total available time for the tour")
    start_lat = st.number_input("Start Latitude", value=41.9028, format="%.4f")
with col2:
    start_time = st.number_input("Start Time (minutes from midnight)", value=540, min_value=0, max_value=1439, help="e.g., 540 = 9:00 AM")
    start_lon = st.number_input("Start Longitude", value=12.4964, format="%.4f")

if st.button("🚀 Generate Tour", type="primary"):
    if not pois_file:
        st.error("Please upload a POIs file")
        st.stop()
    
    if profile_option == "Custom" and not profile_data:
        st.error("Please upload a custom profile JSON")
        st.stop()

    # Prepare request
    files = {'pois_file': pois_file}
    data = {
        'budget': budget,
        'start_time': start_time,
        'start_lat': start_lat,
        'start_lon': start_lon,
    }
    
    if profile_name:
        data['profile_name'] = profile_name
    elif profile_data:
        data['profile_json'] = json.dumps(profile_data)
    
    with st.spinner("Generating tour... This may take a few moments."):
        try:
            response = requests.post("http://localhost:8000/generate_tour", files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # Log the result
                log_entry = {
                    "timestamp": str(pd.Timestamp.now()),
                    "profile": profile_name if profile_name else "custom",
                    "budget": budget,
                    "start_time": start_time,
                    "start_lat": start_lat,
                    "start_lon": start_lon,
                    "result": result
                }
                os.makedirs("data", exist_ok=True)
                with open("data/tour_results", "a", encoding="utf-8") as f:
                    json.dump(log_entry, f, ensure_ascii=False)
                    f.write("\n")
                
                st.success("✅ Tour generated successfully!")
                
                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Score", f"{result['total_score']:.2f}")
                with col2:
                    st.metric("Total Distance", f"{result['total_distance']:.1f} km")
                with col3:
                    st.metric("Total Time", f"{result['total_time']} min")
                
                st.write(f"**Feasible:** {'✅ Yes' if result['is_feasible'] else '❌ No'}")
                
                # Display stops
                st.header("📋 Tour Itinerary")
                if result['stops']:
                    stops_df = pd.DataFrame(result['stops'])
                    stops_df['arrival_time'] = stops_df['arrival'].apply(lambda x: f"{x//60:02d}:{x%60:02d}")
                    stops_df['departure_time'] = stops_df['departure'].apply(lambda x: f"{x//60:02d}:{x%60:02d}")
                    stops_df['wait_min'] = stops_df['wait']
                    
                    st.dataframe(
                        stops_df[['poi_name', 'arrival_time', 'departure_time', 'wait_min', 'travel_distance_km', 'travel_time_min']],
                        width="stretch"
                    )
                else:
                    st.info("No stops in the generated tour")
                    
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the API. Make sure the FastAPI server is running on localhost:8000\nError: {str(e)}")

st.markdown("---")
st.markdown("**Note:** Make sure to start the FastAPI server first: `python app.py` or `uvicorn app:app --reload`")