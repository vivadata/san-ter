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
def load_departements(path: str = "dev2/departements.geojson") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    # on s’assure que le code est bien une chaîne (01, 2A, 971…)
    gdf["code"] = gdf["code"].astype(str)
    return gdf


df = load_rpps()
# Remove any summary/total rows that may be present in the CSV (e.g. "Total général")
# These rows break charts because they contain aggregated totals rather than a department code.
if 'dept_code' in df.columns:
    df = df[~df['dept_code'].astype(str).str.contains('Total', na=False)]
if 'dept_code_clean' in df.columns:
    df = df[~df['dept_code_clean'].astype(str).str.contains('Total', na=False)]
# Nettoyage de la colonne densite (virgules, espaces)
df["densite"] = (
    df["densite"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .str.replace(" ", "", regex=False)
)

df["densite"] = pd.to_numeric(df["densite"], errors="coerce")
gdf_dept = load_departements()

# --- Normaliser les codes de départements dans le GeoDataFrame ---
if 'code' in gdf_dept.columns:
    # nettoyer espaces et assurer string
    gdf_dept['code'] = gdf_dept['code'].astype(str).str.strip()
    # padding pour codes purement numériques (1 -> 01)
    def pad_code(c):
        if c.isdigit():
            return c.zfill(2)
        return c
    gdf_dept['code'] = gdf_dept['code'].apply(pad_code)

    # Cas particulier : certains geojson stockent des codes numériques mais le nom contient 2A/2B
    # corriger en se basant sur la colonne 'nom' si présente
    if 'nom' in gdf_dept.columns:
        mask2a = gdf_dept['nom'].str.contains('2A', na=False)
        mask2b = gdf_dept['nom'].str.contains('2B', na=False)
        gdf_dept.loc[mask2a, 'code'] = '2A'
        gdf_dept.loc[mask2b, 'code'] = '2B'

# Normaliser les codes côté données RPPS
if 'dept_code_clean' in df.columns:
    df['dept_code_clean'] = df['dept_code_clean'].astype(str).str.strip()
    df['dept_code_clean'] = df['dept_code_clean'].str.upper()

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

# Map department codes to their names (if available in the geo dataframe)
if 'code' in gdf_dept.columns and 'nom' in gdf_dept.columns:
    name_map = gdf_dept.set_index('code')['nom'].to_dict()
else:
    name_map = {}

df_densite['dept_name'] = df_densite['dept_code_clean'].map(name_map).fillna(df_densite['dept_code_clean'])

# Afficher le tableau en utilisant les noms de départements
df_display = df_densite.copy()
df_display = df_display.rename(columns={'dept_name': 'Département', 'densite': 'Densité'})
st.dataframe(
    df_display[['Département', 'Densité']].reset_index(drop=True),
    use_container_width=True,
)

# ------------------------------------------------------
# 6. Graphique barres (optionnel mais utile)
# ------------------------------------------------------
st.subheader("Histogramme des densités par département")

# Utiliser le nom du département comme index pour l'histogramme
st.bar_chart(
    df_densite.set_index('dept_name')['densite']
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

# -------------------------------
# Identifier et afficher zones sans / peu de spécialite
# -------------------------------
st.markdown("---")
st.subheader("Départements sans ou à faible densité pour la spécialité sélectionnée")

# Départements pour lesquels il n'y a aucune densité (NaN après jointure)
gdf_all = gdf_merge.copy()
missing_mask = gdf_all["densite"].isna()
missing = gdf_all[missing_mask]

# Seuil bas: percentile réglable (par défaut 10ème percentile)
percentile = st.slider("Seuil faible (percentile)", min_value=1, max_value=50, value=10)
dens_values = gdf_all["densite"].dropna()
if not dens_values.empty:
    low_threshold = float(dens_values.quantile(percentile / 100))
else:
    low_threshold = 0.0

low_mask = gdf_all["densite"].notna() & (gdf_all["densite"] <= low_threshold)
low = gdf_all[low_mask]

st.write(f"Départements sans données : {len(missing)}")
if not missing.empty:
    st.table(missing[["code", "nom"]].rename(columns={"code": "Code", "nom": "Département"}))
else:
    st.write("Aucun département sans donnée pour cette spécialité.")

st.write(f"Départements à faible densité (<= {percentile}ᵉ percentile ≈ {low_threshold:.2f}) : {len(low)}")
if not low.empty:
    st.dataframe(low[["code", "nom", "densite"]].rename(columns={"code": "Code", "nom": "Département", "densite": "Densité"}))
else:
    st.write("Aucun département n'est en-dessous du seuil choisi.")

# Carte dédiée : ajouter couches pour manquants (gris) et faibles (orange) et bulles existantes (rouge)
layers = []

# Bulles existantes (déjà définies)
layer_bubbles = pdk.Layer(
    "ScatterplotLayer",
    data=gdf_bubbles,
    get_position='[lon, lat]',
    get_radius='radius',
    get_fill_color='[200, 30, 30, 160]',
    pickable=True,
    auto_highlight=True,
)
layers.append(layer_bubbles)

if not low.empty:
    low = low.copy()
    low["radius"] = low["densite"].fillna(0) * scale
    layer_low = pdk.Layer(
        "ScatterplotLayer",
        data=low,
        get_position='[lon, lat]',
        get_radius='radius',
        get_fill_color='[255, 165, 0, 200]',  # orange
        pickable=True,
        auto_highlight=True,
    )
    layers.append(layer_low)

if not missing.empty:
    miss = missing.copy()
    # placer un petit cercle fixe pour les manquants
    miss["lon"] = miss.geometry.centroid.x
    miss["lat"] = miss.geometry.centroid.y
    miss["radius"] = 30000
    layer_missing = pdk.Layer(
        "ScatterplotLayer",
        data=miss,
        get_position='[lon, lat]',
        get_radius='radius',
        get_fill_color='[120, 120, 120, 140]',  # gris
        pickable=True,
        auto_highlight=True,
    )
    layers.append(layer_missing)

if layers:
    deck2 = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "Dépt {code} – {nom}\nDensité: {densite}"},
        map_style="mapbox://styles/mapbox/light-v9",
    )
    st.subheader("Carte — départements mis en évidence")
    st.pydeck_chart(deck2)
else:
    st.write("Aucune donnée géographique disponible pour dessiner la carte")

