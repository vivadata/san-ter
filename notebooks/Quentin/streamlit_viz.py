import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import plotly.express as px
from pathlib import Path


st.set_page_config(layout="wide", page_title="Exploration Pathologies & Offres")

BASE = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    # Try multiple likely paths for the datasets
    df_offre = None
    df_med = None
    df_dept = None

    try:
        df_offre = pd.read_csv(BASE / "FV_offre_pathologie_clusters.csv")
    except Exception:
        try:
            df_offre = pd.read_csv(BASE / "offre_pathologie.csv")
        except Exception:
            st.warning("Impossible de charger `FV_offre_pathologie_clusters.csv` ou `offre_pathologie.csv` depuis le dossier du script.")

    try:
        df_med = pd.read_csv(BASE / "FV_medecins_clusters_viz.csv")
    except Exception:
        # Not fatal
        df_med = None

    try:
        df_dept = pd.read_csv(BASE / "pathologie par departement.csv")
    except Exception:
        # try parent folder
        try:
            df_dept = pd.read_csv(BASE.parent / "pathologie par departement.csv")
        except Exception:
            df_dept = None

    # load a departments geojson if present
    gdf_dept = None
    geo_paths = [BASE / "departements-100m.geojson", BASE / "departements.geojson", BASE.parent / "dev2" / "departements.geojson"]
    for p in geo_paths:
        if p.exists():
            try:
                gdf_dept = gpd.read_file(p)
                break
            except Exception:
                gdf_dept = None

    return df_offre, df_med, df_dept, gdf_dept


def find_dept_code_col(gdf, df_dept_codes):
    # Try to find a column in gdf that matches department codes in df_dept_codes
    if gdf is None or df_dept_codes is None:
        return None
    candidates = []
    for c in gdf.columns:
        if gdf[c].dtype == object or pd.api.types.is_integer_dtype(gdf[c]):
            # compare set intersection size
            try:
                common = set(gdf[c].astype(str).str.zfill(2).values) & set(df_dept_codes.astype(str).str.zfill(2).values)
                if len(common) > 0:
                    candidates.append((c, len(common)))
            except Exception:
                continue
    if not candidates:
        return None
    # return best candidate
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def choropleth_layer_from_gdf(gdf, value_col, value_min=None, value_max=None):
    # Prepare geojson-like features for pydeck PolygonLayer
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        try:
            coords = []
            if geom.geom_type == 'MultiPolygon':
                # take exterior of first polygon
                poly = list(geom.geoms)[0]
                coords = [[list(coord) for coord in poly.exterior.coords]]
            else:
                coords = [[list(coord) for coord in geom.exterior.coords]]
        except Exception:
            continue
        v = row.get(value_col, 0) if value_col in row else 0
        features.append({"coordinates": coords[0], "value": float(v) if pd.notna(v) else 0})

    if not features:
        return None

    # normalize values to 0-255 for a red color ramp
    vals = [f["value"] for f in features]
    vmin = value_min if value_min is not None else min(vals)
    vmax = value_max if value_max is not None else max(vals)
    def color_for(v):
        if vmax == vmin:
            t = 0.5
        else:
            t = (v - vmin) / (vmax - vmin)
        r = int(255 * t)
        g = int(60 * (1 - t))
        b = 100
        return [r, g, b, 140]

    layer_data = []
    for f in features:
        layer_data.append({"polygon": f["coordinates"], "value": f["value"], "color": color_for(f["value"])})

    return pdk.Layer(
        "PolygonLayer",
        layer_data,
        get_polygon="polygon",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
        stroked=False,
        extruded=False,
    )


def main():
    st.title("Exploration : Offres & Pathologies — cartes et graphiques")

    df_offre, df_med, df_dept, gdf_dept = load_data()

    st.sidebar.header("Filtres")
    famille_choices = ["(Tous)"]
    if df_offre is not None and 'famille_pathologie' in df_offre.columns:
        famille_choices += sorted(df_offre['famille_pathologie'].dropna().unique().tolist())
    famille = st.sidebar.selectbox("Famille pathologie", famille_choices)

    spec_choices = ["(Tous)"]
    if df_offre is not None and 'specialite' in df_offre.columns:
        spec_choices += sorted(df_offre['specialite'].dropna().unique().tolist())
    specialite = st.sidebar.selectbox("Spécialité", spec_choices)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Affichage des données géographiques et graphiques interactifs.")

    col1, col2 = st.columns((2, 1))

    with col1:
        st.subheader("Carte : points d'offre (médecins / structures)")
        map_df = None
        if df_offre is not None and {'longitude', 'latitude'}.issubset(df_offre.columns):
            map_df = df_offre.copy()
            if famille != "(Tous)":
                map_df = map_df[map_df['famille_pathologie'] == famille]
            if specialite != "(Tous)":
                map_df = map_df[map_df['specialite'] == specialite]
            map_df = map_df.dropna(subset=['longitude', 'latitude'])

        # Try medecins file too if present
        if map_df is None or map_df.empty:
            if df_med is not None and {'longitude', 'latitude'}.issubset(df_med.columns):
                map_df = df_med.copy()
                if 'famille_pathologie' in map_df.columns and famille != "(Tous)":
                    map_df = map_df[map_df['famille_pathologie'] == famille]
                if 'specialite' in map_df.columns and specialite != "(Tous)":
                    map_df = map_df[map_df['specialite'] == specialite]

        if map_df is None or map_df.empty:
            st.info("Aucune donnée de points géographiques trouvée pour ces filtres.")
        else:
            # Use pydeck for interactive layered map
            midpoint = (map_df['latitude'].mean(), map_df['longitude'].mean())
            st.write(f"Points affichés : {len(map_df)}")
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[longitude, latitude]',
                get_radius=100,
                get_fill_color='[200, 30, 0, 160]',
                pickable=True,
            )
            view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=6)
            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text":"{specialite}\n{famille_pathologie}"})
            st.pydeck_chart(r)

    with col2:
        st.subheader("Valeurs agrégées & Top")
        if df_offre is not None:
            agg = df_offre.groupby(['famille_pathologie', 'specialite'], dropna=False).agg({'offre_count': 'sum'}).reset_index()
            if famille != "(Tous)":
                agg = agg[agg['famille_pathologie'] == famille]
            if specialite != "(Tous)":
                agg = agg[agg['specialite'] == specialite]
            top_specs = agg.sort_values('offre_count', ascending=False).head(20)
            fig = px.bar(top_specs, x='offre_count', y='specialite', orientation='h', title='Top spécialités par offre')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune table d'offres chargée (`FV_offre_pathologie_clusters_viz.csv`) pour les agrégations.")

    st.markdown("---")

    st.subheader("Choroplèthe par département")
    if df_dept is None or gdf_dept is None:
        st.info("Fichier départemental ou GeoJSON de départements introuvable dans le dossier du script. Mettez `pathologie par departement.csv` et `departements-100m.geojson` dans le même dossier que ce script.")
    else:
        # use ZONE_0 as code if present
        if 'ZONE_0' in df_dept.columns:
            df_dept['ZONE_0'] = df_dept['ZONE_0'].astype(str).str.zfill(2)
            dept_code_col = find_dept_code_col(gdf_dept, df_dept['ZONE_0'])
            if dept_code_col is None:
                st.warning("Impossible d'identifier automatiquement la colonne de code département dans le GeoJSON. Affichage minimal des données tabulaires.")
                st.dataframe(df_dept.head())
            else:
                gdf_dept = gdf_dept.to_crs(epsg=4326)
                # normalize geojson code col to strings
                gdf_dept[dept_code_col] = gdf_dept[dept_code_col].astype(str).str.zfill(2)
                # pick the famille to merge
                merge_family = famille if famille != "(Tous)" else None
                if merge_family:
                    df_dep_agg = df_dept[df_dept['FAMILLE_PATHOLOGIE'] == merge_family].groupby('ZONE_0', as_index=False).agg({'ind_freq_sum': 'sum'})
                else:
                    df_dep_agg = df_dept.groupby('ZONE_0', as_index=False).agg({'ind_freq_sum': 'sum'})
                merged = gdf_dept.merge(df_dep_agg, left_on=dept_code_col, right_on='ZONE_0', how='left')
                merged['ind_freq_sum'] = merged['ind_freq_sum'].fillna(0)

                # create pydeck polygon layer
                poly_layer = choropleth_layer_from_gdf(merged, 'ind_freq_sum')
                if poly_layer is not None:
                    centroid = merged.geometry.centroid
                    avg_lat = centroid.y.mean()
                    avg_lon = centroid.x.mean()
                    view_state = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=5)
                    deck = pdk.Deck(layers=[poly_layer], initial_view_state=view_state)
                    st.pydeck_chart(deck)
                st.markdown("Top 10 départements par fréquence")
                top_deps = merged[['ZONE_0', 'ind_freq_sum']].sort_values('ind_freq_sum', ascending=False).head(10)
                st.table(top_deps.set_index('ZONE_0'))

    st.markdown("---")
    st.subheader("Aperçu des tables chargées")
    if df_offre is not None:
        st.write("`FV_offre_pathologie_clusters.csv` — aperçu")
        st.dataframe(df_offre.head())
    if df_med is not None:
        st.write("`FV_medecins_clusters_viz.csv` — aperçu")
        st.dataframe(df_med.head())
    if df_dept is not None:
        st.write("`pathologie par departement.csv` — aperçu")
        st.dataframe(df_dept.head())


if __name__ == '__main__':
    main()
