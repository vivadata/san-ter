import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk

# ------------------------------------------------------
# 1. Chargement des données
# ------------------------------------------------------
@st.cache_data
def load_rpps(path: str = "src/rpps_stream/rpps_long_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "dept_code": "string",
            "dept_code_clean": "string",
        }
    )
    return df


@st.cache_data
def load_departements(path: str = "data/departements.geojson") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    # on s’assure que le code est bien une chaîne (01, 2A, 971…)
    gdf["code"] = gdf["code"].astype(str)
    return gdf


df = load_rpps()
# Nettoyage de la colonne densite (virgules, espaces)
df["densite"] = (
    df["densite"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .str.replace(" ", "", regex=False)
)

df["densite"] = pd.to_numeric(df["densite"], errors="coerce")
gdf_dept = load_departements()

# ------------------------------------------------------
# 2. Titre et description
# ------------------------------------------------------
st.title("Densité des médecins par spécialité et département (RPPS)")

st.write(
    "Visualisation de la **densité de médecins** (pour 100 000 habitants) "
    "par **spécialité** et **département** à partir de `rpps_long_clean.csv`."
)

# ------------------------------------------------------
# 3. Menu déroulant des spécialités
# ------------------------------------------------------
specialites = sorted(df["specialite"].dropna().unique().tolist())

selected_specialite = st.selectbox(
    "Choisir une spécialité",
    options=specialites,
    index=0,
)

# ------------------------------------------------------
# 4. Filtrer et agréger par département (densité)
# ------------------------------------------------------
df_filtered = df[df["specialite"] == selected_specialite].copy()

# Si plusieurs lignes par dept, on prend la moyenne de densité
df_densite = (
    df_filtered
    .groupby("dept_code_clean", as_index=False)["densite"]
    .mean()
    .sort_values("dept_code_clean")
)

if df_densite.empty:
    st.warning("Aucune donnée trouvée pour cette spécialité.")
    st.stop()

# ------------------------------------------------------
# 5. Tableau
# ------------------------------------------------------
st.subheader(f"Densité par département – {selected_specialite}")

st.dataframe(
    df_densite.rename(columns={"dept_code_clean": "Département", "densite": "Densité"}),
    use_container_width=True,
)

# ------------------------------------------------------
# 6. Graphique barres (optionnel mais utile)
# ------------------------------------------------------
st.subheader("Histogramme des densités par département")

st.bar_chart(
    df_densite.set_index("dept_code_clean")["densite"]
)

# ------------------------------------------------------
# 7. Carte à bulles
# ------------------------------------------------------
st.subheader("Carte à bulles – Densité par département")

# Jointure avec les départements
# gdf_dept['code'] : code INSEE du département (01, 2A, 75, 971…)
gdf_merge = gdf_dept.merge(
    df_densite,
    left_on="code",
    right_on="dept_code_clean",
    how="left",
)

# On ne garde que les départements pour lesquels on a une densité
gdf_bubbles = gdf_merge.dropna(subset=["densite"]).copy()

# Calcul des centroides pour positionner les bulles
gdf_bubbles["lon"] = gdf_bubbles.geometry.centroid.x
gdf_bubbles["lat"] = gdf_bubbles.geometry.centroid.y

# Échelle du rayon des bulles (à ajuster si besoin)
max_densite = gdf_bubbles["densite"].max()
if max_densite and max_densite > 0:
    scale = 50000 / max_densite
else:
    scale = 1000

gdf_bubbles["radius"] = gdf_bubbles["densite"] * scale

# Préparer les données pour pydeck
bubble_data = gdf_bubbles[["nom", "code", "densite", "lat", "lon", "radius"]]

view_state = pdk.ViewState(
    latitude=46.5,
    longitude=2.5,
    zoom=4.5,
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=bubble_data,
    get_position='[lon, lat]',
    get_radius="radius",
    get_fill_color="[200, 30, 30, 160]",  # rouge transparent
    pickable=True,
    auto_highlight=True,
)

tooltip = {
    "text": "Département {code} – {nom}\nDensité : {densite} médecins / 100 000 hab."
}

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/light-v9",
)

st.pydeck_chart(deck)

