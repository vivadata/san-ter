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
st.subheader("Carte — densité par département (rouge = faible, bleu = élevé)")

# Jointure avec les départements (pour obtenir géométrie + nom)
gdf_merge = gdf_dept.merge(
    df_densite,
    left_on="code",
    right_on="dept_code_clean",
    how="left",
)

# Préparer une couleur par département : rouge (faible) → bleu (élevé)
dens_min = float(gdf_merge['densite'].min(skipna=True)) if gdf_merge['densite'].notna().any() else 0.0
dens_max = float(gdf_merge['densite'].max(skipna=True)) if gdf_merge['densite'].notna().any() else 1.0

def dens_to_color(v, vmin=dens_min, vmax=dens_max):
    # NaN -> gris
    try:
        if pd.isna(v):
            return [200, 200, 200, 120]
    except Exception:
        return [200, 200, 200, 120]
    if vmax <= vmin:
        t = 0.5
    else:
        t = (float(v) - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
    r = int(255 * (1 - t))
    g = int(30 * (1 - abs(0.5 - t) * 2))
    b = int(255 * t)
    return [r, g, b, 180]

gdf_merge['fill_color'] = gdf_merge['densite'].apply(dens_to_color)

# Convertir en GeoJSON via __geo_interface__ pour pydeck
geojson = gdf_merge.__geo_interface__

view_state = pdk.ViewState(latitude=46.5, longitude=2.5, zoom=4.5)

layer = pdk.Layer(
    "GeoJsonLayer",
    data=geojson,
    stroked=False,
    filled=True,
    get_fill_color='properties.fill_color',
    pickable=True,
    auto_highlight=True,
)

tooltip = {"text": "Département {properties.code} – {properties.nom}\nDensité: {properties.densite}"}

deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v9")
st.pydeck_chart(deck)

# Petite légende
st.markdown(
    """
    **Légende**: <span style='color:#ff0000'>⬤</span> faible densité — <span style='color:#0000ff'>⬤</span> forte densité — <span style='color:#a0a0a0'>⬤</span> pas de donnée
    """,
    unsafe_allow_html=True,
)

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

# (La carte choroplèthe ci-dessus remplace les cartes à bulles et les couches séparées.)

