import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Carte pathologies et médecins")

# -----------------------------
# 1. Chargement des données
# -----------------------------
@st.cache_data
def load_data():
    patho_dept = gpd.read_file("notebooks/Quentin/FV_pathologie_departements.gpkg")
    offre_clusters = gpd.read_file("notebooks/Quentin/FV_offre_pathologie_clusters.gpkg")
    med = pd.read_csv("notebooks/Quentin/offre_medical.csv")
    med_gdf = gpd.GeoDataFrame(
        med,
        geometry=gpd.points_from_xy(med.longitude, med.latitude),
        crs="EPSG:4326"
    )
    return patho_dept, offre_clusters, med_gdf

patho_dept, offre_clusters, med_gdf = load_data()

# -----------------------------
# 2. Menu pathologie
# -----------------------------
pathologies = sorted(patho_dept["FAMILLE_PATHOLOGIE"].dropna().unique())
choix_patho = st.selectbox("Choisir une pathologie :", pathologies)

# Filtrer les données
dept_sel = patho_dept[patho_dept["FAMILLE_PATHOLOGIE"] == choix_patho]
clusters_sel = offre_clusters[offre_clusters["FAMILLE_PATHOLOGIE"] == choix_patho]
med_sel = med_gdf[med_gdf["Libellé savoir-faire"].str.contains(choix_patho, na=False)]

# -----------------------------
# 3. Carte Folium
# -----------------------------
m = folium.Map(location=[46.6, 2.5], zoom_start=6, tiles="cartodbpositron")

# --- Départements (demande) ---
min_d = dept_sel["ind_freq_sum"].min()
max_d = dept_sel["ind_freq_sum"].max()

def color_dept(v):
    ratio = (v - min_d) / (max_d - min_d + 1e-5)
    r = int(255 * (1 - ratio))
    g = int(255 * ratio)
    return f"#{r:02x}{g:02x}00"

for _, row in dept_sel.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature, val=row["ind_freq_sum"]: {
            "fillColor": color_dept(val),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5
        },
        tooltip=f"{row['ZONE_0']} - {row['FAMILLE_PATHOLOGIE']}: {row['ind_freq_sum']}"
    ).add_to(m)

# --- Clusters (offre) ---
min_c = clusters_sel["med_count"].min()
max_c = clusters_sel["med_count"].max()

def color_cluster(v):
    ratio = (v - min_c) / (max_c - min_c + 1e-5)
    r = int(255 * (1 - ratio))
    g = int(255 * ratio)
    return f"#{r:02x}{g:02x}00"

for _, row in clusters_sel.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature, val=row["med_count"]: {
            "fillColor": color_cluster(val),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.6
        },
        tooltip=f"Cluster: {row['cluster_id']}, Médecins: {row['med_count']}"
    ).add_to(m)

# --- Médecins (points) ---
marker_cluster = MarkerCluster(name="Médecins").add_to(m)
for _, row in med_sel.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        color="blue",
        fill=True,
        fill_opacity=0.9,
        popup=f"<b>Spécialité:</b> {row['Libellé savoir-faire']}<br>"
              f"<b>Adresse:</b> {row.get('result_city','')}"
    ).add_to(marker_cluster)

# --- Légende et calques ---
folium.LayerControl().add_to(m)

# -----------------------------
# 4. Affichage Streamlit
# -----------------------------
st.title(f"Carte des pathologies et médecins : {choix_patho}")
st_folium(m, width=1200, height=800)