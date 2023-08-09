import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from st_clickable_images import clickable_images
import os
import base64


#  Load and transform data
data = pd.read_csv("data/ai_ready/x-ai_data.csv")
data = data[data["class"] == 1]
data["age"] = np.random.randint(0, 20, len(data))
data["lat"] = np.random.randn(len(data), 1) / 2 + 42.032974
data["lon"] = np.random.randn(len(data), 1) / 2 + -93.581543
data["coordinates"] = data.apply(lambda row: f"{row.lat:.2f} / {row.lon:.2f}", axis=1)

# Main title
st.markdown(
    f"<h1 style='text-align: center; color: white;'>Silos lifetime map</h1>",
    unsafe_allow_html=True,
)

# Slider
x = st.slider("Silos that are older than", 0, 20)

# Map
layer = pdk.Layer(
    "ColumnLayer",  # `type` positional argument is here
    data[data.age >= x],
    get_position=["lon", "lat"],
    get_elevation="age",
    auto_highlight=True,
    radius=500,
    elevation_scale=500,
    pickable=True,
    extruded=True,
    coverage=1,
)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=-93.581543,
    latitude=42.032974,
    zoom=7,
    min_zoom=5,
    max_zoom=15,
    pitch=40.5,
    bearing=-27.36,
)

# Combined all of it and render a viewport
st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="road",
        tooltip={
            "html": "<b>Age:</b> {age}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        },
    )
)

st.markdown(
    f"<p style='text-align: center; color: white;'>The higher the point, the older the silo.</p>",
    unsafe_allow_html=True,
)

# Images part
st.markdown(
    f"<h1 style='text-align: center; color: white;'>Silos over {x} years</h1>",
    unsafe_allow_html=True,
)


images = []
images_path = "data/ai_ready/images"
sample_images = data[data.age >= x].filename

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
