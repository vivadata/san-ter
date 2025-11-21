import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# -----------------------------
# 1. Configuration Streamlit
# -----------------------------
st.set_page_config(layout="wide", page_title="Carte des offres par pathologie")

# -----------------------------
# 2. Charger les fichiers GPKG
# -----------------------------
GPKG_CLUSTERS = "notebooks/Quentin/iris_clusters_200.gpkg"
GPKG_OFFRE = "notebooks/Quentin/offre_by_cluster_patho.gpkg"

# Charger les IRIS et fusionner les IRIS par cluster_id
gdf_iris = gpd.read_file(GPKG_CLUSTERS)
clusters_gdf = gdf_iris.dissolve(by='cluster_id', as_index=False)[['cluster_id', 'geometry']]

# Charger les offres/pathologies
offre_gdf = gpd.read_file(GPKG_OFFRE)
offre_gdf.columns = [c.lower() for c in offre_gdf.columns]

# -----------------------------
# 3. Sélecteur de pathologie
# -----------------------------
pathologies = sorted(offre_gdf['famille_pathologie'].dropna().unique())
choix_patho = st.selectbox("Choisir une pathologie :", pathologies)

# Sélectionner les offres pour la pathologie choisie
offre_sel = offre_gdf[offre_gdf['famille_pathologie'] == choix_patho][['cluster_id', 'specialist_count']]

# -----------------------------
# 4. Merge pour garder tous les clusters
# -----------------------------
gdf_sel = clusters_gdf.merge(
    offre_sel,
    on='cluster_id',
    how='left'  # garde tous les 200 clusters
)

# Remplacer les NaN par 0 pour les clusters sans spécialistes
gdf_sel['specialist_count'] = gdf_sel['specialist_count'].fillna(0)

# Diagnostique des géométries manquantes / vides
try:
    missing_geom_mask = gdf_sel['geometry'].isna()
except Exception:
    # fallback if isna doesn't work for geometry column
    missing_geom_mask = gdf_sel['geometry'].apply(lambda g: g is None)
empty_geom_mask = gdf_sel['geometry'].apply(lambda g: getattr(g, 'is_empty', False) if g is not None else True)
num_missing = int(missing_geom_mask.sum())
num_empty = int(empty_geom_mask.sum())
if num_missing or num_empty:
    st.warning(f"Clusters sans géométrie: {num_missing}, géométries vides: {num_empty} — elles seront ignorées à l'affichage.")

# Garder seulement les lignes avec géometrie non nulle et non vide
valid_mask = (~missing_geom_mask) & (~empty_geom_mask)
gdf_sel = gdf_sel[valid_mask].copy()

st.write(f"Nombre de clusters affichés pour {choix_patho} :", len(gdf_sel))

# -----------------------------
# 5. Carte Folium
# -----------------------------
m = folium.Map(location=[46.6, 2.5], zoom_start=6, tiles="cartodbpositron")

# Fonction pour colorer selon specialist_count
min_count = gdf_sel['specialist_count'].min()
max_count = gdf_sel['specialist_count'].max()

def color_cluster(v):
    ratio = (v - min_count) / (max_count - min_count + 1e-5)
    r = int(255 * (1 - ratio))
    g = int(255 * ratio)
    b = 0
    return f"#{r:02x}{g:02x}{b:02x}"

# Ajouter les clusters fusionnés sur la carte
for _, row in gdf_sel.iterrows():
    geom = row.get('geometry') if isinstance(row, dict) else getattr(row, 'geometry', None)
    if geom is None:
        # sécurité supplémentaire — ne pas passer None à folium
        continue
    if getattr(geom, 'is_empty', False):
        continue
    # construire tooltip en string
    tooltip_txt = f"Cluster: {row.get('cluster_id', '')} — Spécialistes: {int(row.get('specialist_count', 0))}"
    folium.GeoJson(
        geom,
        style_function=lambda feature, val=row['specialist_count']: {
            'fillColor': color_cluster(val),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        },
        tooltip=tooltip_txt
    ).add_to(m)

folium.LayerControl().add_to(m)

# -----------------------------
# 6. Affichage dans Streamlit
# -----------------------------
st.title(f"Carte des offres de soins : {choix_patho}")
st_folium(m, width=1200, height=800)