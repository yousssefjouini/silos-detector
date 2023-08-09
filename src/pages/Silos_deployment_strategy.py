import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from st_clickable_images import clickable_images
import os
import base64
from geopy.geocoders import Nominatim
from sklearn.neighbors import BallTree
import numpy as np
from typing import Tuple


def closest_neighbor(silos: pd.DataFrame, search: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Choose closest silos from search place
    """

    silos_gps = silos[["lat", "lon"]].values
    search_gps = search[["lat", "lon"]].values
    tree = BallTree(search_gps, leaf_size=15, metric="haversine")
    distance, index = tree.query(silos_gps, k=1)
    earth_radius = 6371
    distance_in_km = distance * earth_radius

    neigh = pd.DataFrame(
        {
            "filename": silos.filename,
            "lat": silos.lat,
            "lon": silos.lon,
            "dist": distance_in_km[:, 0],
        }
    )
    return neigh.sort_values(by="dist")[:n]


def location_to_coord(place: str) -> Tuple[float, float]:
    """
    Transform location into coordinates
    """

    geolocator = Nominatim(user_agent="http")
    location = geolocator.geocode(place)
    return location.latitude, location.longitude


# Load and transform data
data = pd.read_csv("data/ai_ready/x-ai_data.csv")
data["age"] = np.random.randint(0, 30)
data["lat"] = np.random.randn(len(data), 1) / 2 + 42.032974
data["lon"] = np.random.randn(len(data), 1) / 2 + -93.581543
data["coordinates"] = data.apply(lambda row: f"{row.lat:.2f} / {row.lon:.2f}", axis=1)
data["coord"] = data.apply(lambda row: [row.lat, row.lon], axis=1)

# Main title
st.markdown(
    f"<h1 style='text-align: center; color: white;'>Silos location strategy</h1>",
    unsafe_allow_html=True,
)

# Search bar
title = st.text_input("Location", "Des Moines")

# Searched city info
lat, long = location_to_coord(title)
df = pd.DataFrame(columns=["lon", "lat"])
df.loc[0] = [long, lat]

# Neighbors slider
x = st.slider("How many silos near the location you want to display", 0, 20)

# Map
layer = pdk.Layer(
    "HexagonLayer",  # `type` positional argument is here
    df,
    get_position=["lon", "lat"],
    auto_highlight=True,
    elevation_scale=1,
    radius=600,
    pickable=True,
    elevation_range=[0, 1],
    extruded=True,
    coverage=1,
)


layer_silo = pdk.Layer(
    "ColumnLayer",  # `type` positional argument is here
    closest_neighbor(data[data["class"] == 1], df, x),
    get_position=["lon", "lat"],
    auto_highlight=True,
    elevation_scale=1,
    radius=200,
    pickable=True,
    elevation_range=[0, 1],
    extruded=True,
    coverage=1,
)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=long,
    latitude=lat,
    zoom=11,
    min_zoom=5,
    max_zoom=15,
    pitch=40.5,
    bearing=-27.36,
)

# Combined all of it and render a viewport
st.pydeck_chart(
    pdk.Deck(
        layers=[layer, layer_silo], initial_view_state=view_state, map_style="road"
    )
)

# Images partÒ
st.markdown(
    f"<h2 style='text-align: center; color: white;'>{x} closest silos</h2>",
    unsafe_allow_html=True,
)

images = []
images_path = "data/ai_ready/images"
sample_images = closest_neighbor(data[data["class"] == 1], df, x).filename

if len(sample_images) > 100:
    sample_images = sample_images[:99]
else:
    pass

for file in [os.path.join(images_path, sample_image) for sample_image in sample_images]:
    with open(file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
        images.append(f"data:image/jpeg;base64,{encoded}")

clicked = clickable_images(
    images,
    titles=[f"Image #{str(i)}" for i in range(2)],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "15px", "height": "200px"},
)
