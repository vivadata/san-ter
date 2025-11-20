import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Carte médecins")

# -----------------------------
# 1. Chargement des données
# -----------------------------
@st.cache_data
def load_data():
    clusters = gpd.read_file("notebooks/Quentin/FV_medecins_clusters_viz_simplifie.gpkg")
    med = pd.read_csv("notebooks/Quentin/offre_medical.csv")
    # convertir médecins en GeoDataFrame
    med_gdf = gpd.GeoDataFrame(
        med,
        geometry=gpd.points_from_xy(med.longitude, med.latitude),
        crs="EPSG:4326"
    )
    return clusters, med_gdf

clusters, med_gdf = load_data()

# -----------------------------
# 2. Menu spécialité
# -----------------------------
specialites = sorted(clusters["Libellé savoir-faire"].dropna().unique())
choix_specialite = st.selectbox("Choisir une spécialité :", specialites)

# Filtrer clusters et médecins
clusters_sel = clusters[clusters["Libellé savoir-faire"] == choix_specialite]
med_sel = med_gdf[med_gdf["Libellé savoir-faire"] == choix_specialite]

# -----------------------------
# 3. Carte Folium
# -----------------------------
m = folium.Map(location=[46.6, 2.5], zoom_start=6, tiles="cartodbpositron")

# --- Définir couleur en fonction de la densité ---
min_c = clusters_sel["med_count"].min()
max_c = clusters_sel["med_count"].max()

def color_scale(v):
    if pd.isna(v):
        v = 0
    ratio = (v - min_c) / (max_c - min_c + 1e-5)
    # couleur du vert clair au vert foncé
    r = int(255 * (1 - ratio))
    g = int(255 * ratio)
    b = 0
    return f"#{r:02x}{g:02x}{b:02x}"

# --- Ajouter les polygones des clusters ---
for _, row in clusters_sel.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda feature, count=row["med_count"]: {
            'fillColor': color_scale(count),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=f"Cluster: {row['cluster_id']}, Médecins: {row['med_count']}"
    ).add_to(m)

# --- Ajouter les points exacts des médecins ---
marker_cluster = MarkerCluster(name="Médecins").add_to(m)
for _, row in med_sel.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        color="blue",
        fill=True,
        fill_opacity=0.9,
        popup=folium.Popup(
            f"<b>Médecin:</b> {row['Libellé savoir-faire']}<br>"
            f"<b>Adresse:</b> {row.get('result_city','')}",
            max_width=250
        )
    ).add_to(marker_cluster)

# --- Légende et contrôle de calques ---
folium.LayerControl().add_to(m)

# -----------------------------
# 4. Affichage Streamlit
# -----------------------------
st.title(f"Carte des médecins : {choix_specialite}")
st_folium(m, width=1200, height=800)