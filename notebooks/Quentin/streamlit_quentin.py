import streamlit as st
from pathlib import Path
import pandas as pd
import io
import os
import pydeck as pdk
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quentin — Explorer datasets", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

st.title("📁 Quentin — Explorateur de jeux de données")
st.write("Explorez rapidement les CSV et fichiers géographiques présents dans le dossier `notebooks/Quentin`. Interface simple: aperçu, graphiques rapides, carte interactive et téléchargement.")


@st.cache_data
def list_files(folder: Path):
    files = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name))
    return files


@st.cache_data
def load_csv(path: Path, nrows=None):
    try:
        if nrows:
            return pd.read_csv(path, nrows=nrows)
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Erreur lecture CSV: {e}")
        return None


@st.cache_data
def load_geofile(path: Path, layer=None):
    try:
        import geopandas as gpd
    except Exception:
        st.error("La librairie 'geopandas' est requise pour ouvrir des fichiers géographiques.")
        return None
    try:
        if layer:
            gdf = gpd.read_file(path, layer=layer)
        else:
            gdf = gpd.read_file(path)
        return gdf
    except Exception as e:
        st.error(f"Erreur lecture géofichier: {e}")
        return None


def colorize_series(vals, cmap_name='RdYlBu_r', vmin=None, vmax=None, alpha=180):
    try:
        import matplotlib
        cmap = matplotlib.cm.get_cmap(cmap_name)
        if vmin is None:
            vmin = np.nanmin(vals)
        if vmax is None:
            vmax = np.nanmax(vals)
        def mapv(x):
            try:
                if np.isnan(x):
                    return [200,200,200,120]
            except Exception:
                return [200,200,200,120]
            t = 0.0 if vmax == vmin else (float(x)-vmin)/(vmax-vmin)
            t = max(0.0, min(1.0, t))
            rgba = cmap(t)
            return [int(255*rgba[0]), int(255*rgba[1]), int(255*rgba[2]), int(alpha)]
        return [mapv(x) for x in vals]
    except Exception:
        return [[200,200,200,120] for _ in vals]


files = list_files(BASE_DIR)

col_files, col_preview = st.columns([1, 3])

with col_files:
    st.subheader("Fichiers")
    selectable = [f.name for f in files if f.is_file()]
    selected = st.selectbox("Choisir un fichier", options=selectable)
    st.markdown("---")
    st.write(f"Dossier: `{BASE_DIR}`")
    st.write(f"Nombre de fichiers: {len(selectable)}")
    if st.button("Rafraîchir"):
        files = list_files(BASE_DIR)
        st.experimental_rerun()

with col_preview:
    if not selected:
        st.info("Sélectionnez un fichier à gauche.")
    else:
        sel_path = BASE_DIR / selected
        st.subheader(selected)
        st.write(f"Taille: {sel_path.stat().st_size:,} bytes")

        ext = sel_path.suffix.lower()

        # --- CSV ---
        if ext in ['.csv', '.txt']:
            n_preview = st.number_input("Lignes aperçu", min_value=5, max_value=2000, value=200)
            df = load_csv(sel_path, nrows=int(n_preview))
            if df is not None:
                st.markdown("**Aperçu**")
                st.dataframe(df)
                st.markdown("**Colonnes & types**")
                st.write(df.dtypes.astype(str).to_frame('dtype'))
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if num_cols:
                    with st.expander("Graphiques rapides"):
                        col = st.selectbox("Colonne numérique", options=num_cols)
                        st.bar_chart(df[col].dropna().sort_values(ascending=False).head(50))
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button("Télécharger aperçu CSV", csv_bytes, file_name=f"preview-{selected}")

        # --- Geo files (GeoJSON, Shapefile, GPKG) ---
        elif ext in ['.geojson', '.json', '.shp', '.gpkg']:
            st.markdown("**Fichier géographique**")

            # If gpkg ask layer
            layer = None
            if ext == '.gpkg':
                try:
                    import fiona
                    layers = fiona.listlayers(str(sel_path))
                    layer = st.selectbox('Choisir une couche (layer) du GPKG', options=['(par défaut)'] + layers)
                    if layer == '(par défaut)':
                        layer = None
                except Exception:
                    st.info('Impossible de lister les couches GPkg (fiona non disponible). Tentative de lecture par défaut.')

            gdf = load_geofile(sel_path, layer=layer)
            if gdf is not None:
                st.write(f"Géométries: {len(gdf)} — colonnes: {list(gdf.columns)}")
                st.dataframe(gdf.head(50))

                # Option choropleth
                numeric_cols = gdf.select_dtypes(include=['number']).columns.tolist()
                chosen = None
                if numeric_cols:
                    chosen = st.selectbox('Colorer par (colonne numérique)', options=['(aucune)'] + numeric_cols)
                    if chosen == '(aucune)':
                        chosen = None

                opacity = st.slider('Opacité couches', min_value=10, max_value=255, value=180)

                # Prepare colors and geojson
                try:
                    gdf_display = gdf.copy()
                    if 'geometry' in gdf_display.columns:
                        try:
                            gdf_display = gdf_display.to_crs(epsg=4326)
                        except Exception:
                            pass
                    if chosen:
                        vals = gdf_display[chosen].values
                        gdf_display['fill_color'] = colorize_series(vals, alpha=opacity)
                    else:
                        gdf_display['fill_color'] = [[200,200,200,120] for _ in range(len(gdf_display))]

                    geojson = gdf_display.__geo_interface__
                    # center view
                    try:
                        centroid = gdf_display.geometry.unary_union.centroid
                        view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=6)
                    except Exception:
                        view_state = pdk.ViewState(latitude=46.5, longitude=2.5, zoom=5)

                    layer = pdk.Layer(
                        "GeoJsonLayer",
                        data=geojson,
                        stroked=True,
                        filled=True,
                        get_fill_color='properties.fill_color',
                        get_line_color=[80,80,80],
                        pickable=True,
                        auto_highlight=True,
                    )
                    tooltip = {"text": "{properties} -> cliquez pour plus"}
                    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
                    st.pydeck_chart(deck)

                    # small colorbar if chosen
                    if chosen:
                        try:
                            fig, ax = plt.subplots(figsize=(6, 0.4))
                            cmap = plt.get_cmap('RdYlBu_r')
                            norm = plt.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
                            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
                            cb.set_label(chosen)
                            st.pyplot(fig)
                        except Exception:
                            pass

                except Exception as e:
                    st.error(f"Erreur préparation carte: {e}")

                # allow download attributes
                try:
                    buf = io.BytesIO()
                    gdf.drop(columns='geometry').to_csv(buf, index=False)
                    st.download_button("Télécharger attributs (CSV)", buf.getvalue(), file_name=f"{selected}.attributes.csv")
                except Exception:
                    st.write("Impossible d'exporter les attributs.")

        # --- Notebooks ---
        elif ext == '.ipynb':
            st.markdown("Notebook Jupyter détecté — ouvrez-le dans votre éditeur pour l'exécuter.")
            st.write(f"Chemin: `{sel_path}`")

        else:
            st.info("Type non géré automatiquement. Téléchargement disponible ci‑dessous.")
            with open(sel_path, 'rb') as fh:
                data = fh.read()
            st.download_button("Télécharger fichier", data, file_name=selected)

        # raw download
        try:
            with open(sel_path, 'rb') as fh:
                data = fh.read()
            st.download_button("Télécharger fichier complet", data, file_name=selected)
        except Exception:
            pass


st.sidebar.markdown("---")
st.sidebar.write("Dossier: " + str(BASE_DIR))
st.sidebar.markdown("**Fichiers rapides**")
for p in files:
    if p.is_file():
        st.sidebar.write(p.name)

st.caption("UI simple: aperçu, carte interactive, téléchargements. Dites-moi si vous voulez d'autres vues (comparaison, filtres).")
