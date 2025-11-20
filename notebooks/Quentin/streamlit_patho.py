import streamlit as st
import pandas as pd
import geopandas as gpd
try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except Exception:
    st_folium = None
    HAS_ST_FOLIUM = False
import folium

st.title("Analyse de l'offre de soins")

st.set_page_config(page_title="Projet Quentin - Santé & Territoires",
                   layout="wide")


# ---------- FONCTIONS DE CHARGEMENT ----------

@st.cache_data
def load_offre_medicale():
    # adapte sep=";" ou "," selon tes fichiers
    return pd.read_csv("offre_medical.csv", sep=";")

@st.cache_data
def load_offre_pathologie():
    return pd.read_csv("offre_pathologie.csv", sep=";")

@st.cache_data
def load_iris_clusters():
    return gpd.read_file("iris_clusters_200.gpkg")

# ---------- SIDEBAR ----------

page = st.sidebar.selectbox(
    "Choisir une vue",
    [
        "Offre médicale (table + graph)",
        "Offre par pathologie",
        "Carte des clusters IRIS"
    ]
)

# ---------- PAGES ----------

if page == "Offre médicale (table + graph)":
    df = load_offre_medicale()
    st.subheader("Table des données – offre médicale")
    st.dataframe(df)

    # Exemple de graph : nombre de médecins par département
    if "dept_code" in df.columns:
        st.subheader("Effectif par département")
        agg = df.groupby("dept_code").sum(numeric_only=True)
        st.bar_chart(agg)

elif page == "Offre par pathologie":
    df = load_offre_pathologie()
    st.subheader("Table des données – offre par pathologie")
    st.dataframe(df)

    # Exemple : nombre total par pathologie
    if "pathologie" in df.columns:
        st.subheader("Effectif par pathologie")
        agg = df.groupby("pathologie").sum(numeric_only=True)
        st.bar_chart(agg)

elif page == "Carte des clusters IRIS":
    gdf = load_iris_clusters()

    st.subheader("Carte des clusters IRIS")

    # centre de la carte
    gdf = gdf.to_crs(epsg=4326)  # au cas où
    center = [gdf.geometry.centroid.y.mean(),
              gdf.geometry.centroid.x.mean()]

    m = folium.Map(location=center, zoom_start=8)

    # si tu as une colonne "cluster", on l’affiche dans le popup
    folium.GeoJson(
        gdf,
        tooltip=folium.GeoJsonTooltip(fields=gdf.columns[:5].tolist())
    ).add_to(m)

    if HAS_ST_FOLIUM:
        # affichage via streamlit_folium si dispo
        st_folium(m, width=900, height=600)
    else:
        # fallback: informer l'utilisateur et afficher un aperçu par centroïdes
        st.warning("Le package 'streamlit_folium' n'est pas disponible : affichage Folium désactivé. Affichage alternatif par centroïdes.")
        try:
            gdf_wgs = gdf.to_crs(epsg=4326)
            centroids = gdf_wgs.geometry.centroid
            df_points = pd.DataFrame({
                'lat': centroids.y,
                'lon': centroids.x,
            })
            st.map(df_points)
        except Exception as e:
            st.error(f"Impossible d'afficher la carte de remplacement : {e}")
