"""
Streamlit app for visualizing safety incidents logged by the Monitoring system. 
This dashboard displays key metrics, a log of incidents, and recent images of violations. 
"""


import pandas as pd
import streamlit as st

from src.config import AppConfig


INCIDENT_COLUMNS = [
    "timestamp",
    "person_id",
    "violation_type",
    "status",
    "image_path"
]


# Set page configuration
st.set_page_config(page_title="Industrial Safety Monitor", layout="wide")

config = AppConfig()

# Title of the dashboard
st.title("Industrial Safety Monitoring Dashboard")

# Load incident data from CSV log file
if config.incident_log_path.exists():
    df = pd.read_csv(config.incident_log_path)
else:
    df = pd.DataFrame(columns=INCIDENT_COLUMNS)

# Display key metrics in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Incidents", len(df))

with col2:
    no_helmet = len(
        df[df["violation_type"] == "No Helmet"]
    )

    st.metric("No Helmet", no_helmet)

with col3:
    zone = len(
        df[df["violation_type"] == "Restricted Zone"]
    )

    st.metric("Zone Violations", zone)


# Display incident log in a table
st.subheader("Incident Log")

st.dataframe(df)

# Display recent incident images
st.subheader("Recent Incident Images")

# Get all images in the directory sorted by creation time
if config.incident_image_dir.exists():
    images = sorted(config.incident_image_dir.glob("*.jpg"), reverse=True)
else:
    images = []

# Show the 5 most recent images
for img in images[:5]:

    st.image(str(img), width=300, caption=img.name)
