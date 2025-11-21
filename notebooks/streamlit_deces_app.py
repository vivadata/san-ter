import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Décès France 2018-2023 — Dashboard")


@st.cache_data
def load_data(csv_path):
    # read as strings to handle french decimals and mixed quoting
    df = pd.read_csv(csv_path, dtype=str)

    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Clean 'valeur' column: replace comma decimal with dot and coerce to float
    if 'valeur' in df.columns:
        df['valeur'] = df['valeur'].replace('', np.nan)
        df['valeur'] = df['valeur'].astype(str).str.replace('\xa0', '', regex=False)
        df['valeur'] = df['valeur'].str.replace(',', '.')
        df['valeur'] = pd.to_numeric(df['valeur'], errors='coerce')

    # Ensure departement code is string and trimmed
    if 'departement_de_domicile' in df.columns:
        df['departement_de_domicile'] = df['departement_de_domicile'].astype(str).str.strip()

    # Cast year
    if 'annee_de_deces' in df.columns:
        df['annee_de_deces'] = pd.to_numeric(df['annee_de_deces'], errors='coerce').astype('Int64')

    return df


def try_load_geojson(path):
    try:
        import geopandas as gpd
    except Exception:
        return None, "geopandas non installé"

    try:
        gdf = gpd.read_file(path)
        return gdf, None
    except Exception as e:
        return None, str(e)


def detect_geo_code_col(gdf):
    # find a column likely containing department codes
    candidates = [c for c in gdf.columns if 'code' in c.lower() or 'dep' in c.lower() or 'insee' in c.lower()]
    if len(candidates) == 0:
        # fallback to first non-geometry column
        for c in gdf.columns:
            if c != gdf.geometry.name:
                return c
    return candidates[0]


def main():
    st.title("Décès France 2018-2023 — Dashboard")

    csv_path = st.sidebar.text_input("Chemin du fichier CSV", "notebooks/data deces 2018-2023 - Données.csv")
    geo_path = st.sidebar.text_input("Chemin GeoJSON (départements)", "dev2/departements.geojson")

    df = load_data(csv_path)

    if df is None or df.shape[0] == 0:
        st.error("Impossible de charger les données CSV. Vérifiez le chemin et le format.")
        return

    # filters
    years = sorted(df['annee_de_deces'].dropna().unique().astype(int).tolist())
    year = st.sidebar.selectbox("Année", years, index=len(years)-1)
    sex_options = ['Tous'] + sorted(df['sexe'].dropna().unique().tolist())
    sexe = st.sidebar.selectbox("Sexe", sex_options)

    # fixed animation speed: 5000 ms (5 secondes) par année
    speed_ms = 5000

    # subset
    q = df[df['annee_de_deces'] == year]
    if sexe != 'Tous':
        q = q[q['sexe'] == sexe]

    # KPI calculations
    total_obs = len(q)
    avg_valeur = q['valeur'].mean()

    # department with most deaths according to the 'valeur' column (sum)
    if 'departement_de_domicile' in q.columns and 'valeur' in q.columns and q['departement_de_domicile'].dropna().size > 0:
        s = q.groupby('departement_de_domicile', dropna=True)['valeur'].sum()
        if s.size > 0:
            dept_most = s.idxmax()
            dept_most_count = s.max()
        else:
            dept_most = None
            dept_most_count = None
    else:
        dept_most = None
        dept_most_count = None

    k1, k2, k3 = st.columns(3)
    k1.metric("Observations (année)", f"{total_obs:n}")
    k2.metric("Valeur moyenne (taux)", f"{avg_valeur:.2f}" if not np.isnan(avg_valeur) else "N/A")
    k3.metric("departement ou il y a eu le plus de décès", f"{dept_most} ({dept_most_count:.2f})" if dept_most is not None else "N/A")

    st.markdown("---")

    # Bar charts
    st.subheader("Bar charts")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Évolution 2018–2023 — top 5 départements (somme de 'valeur')")
        if 'departement_de_domicile' in df.columns and 'valeur' in df.columns and 'annee_de_deces' in df.columns:
            # respect current sexe filter but use all years to show trend
            df_sex = df if sexe == 'Tous' else df[df['sexe'] == sexe]

            # aggregate sum per year and department
            agg = df_sex.groupby(['annee_de_deces', 'departement_de_domicile'], dropna=True)['valeur'].sum().reset_index()

            # find top 5 departments by total across all years (within selected sexe)
            top5 = agg.groupby('departement_de_domicile')['valeur'].sum().nlargest(5).index.tolist()

            plot_df = agg[agg['departement_de_domicile'].isin(top5)].copy()
            plot_df['annee_de_deces'] = plot_df['annee_de_deces'].astype(str)

            fig = px.line(
                plot_df,
                x='annee_de_deces',
                y='valeur',
                color='departement_de_domicile',
                markers=True,
                labels={'valeur': 'Somme de valeur', 'annee_de_deces': 'Année', 'departement_de_domicile': 'Département'},
                height=450
            )
            fig.update_layout(yaxis_title='Somme de valeur', xaxis=dict(dtick=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Colonnes nécessaires (`departement_de_domicile`, `valeur`, `annee_de_deces`) introuvables')

    with col2:
        st.write("Par cause initiale de décès (top 5)")
        if 'cause_initiale_de_deces' in q.columns:
            g2 = q.groupby('cause_initiale_de_deces')['valeur'].mean().reset_index().sort_values('valeur', ascending=False).head(5)
            fig2 = px.bar(g2, x='valeur', y='cause_initiale_de_deces', orientation='h', labels={'valeur':'Taux moyen','cause_initiale_de_deces':'Cause'}, height=450)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info('Colonne `cause_initiale_de_deces` introuvable')

    st.markdown("---")

    # Map
    st.subheader("Carte choroplèthe par département")
    st.write("La carte montre la moyenne du taux (`valeur`) par département pour la sélection actuelle.")

    gdf, err = try_load_geojson(geo_path)
    if gdf is None:
        st.error(f"Impossible de charger le GeoJSON: {err}. Installez `geopandas` ou corrigez le chemin.")
        return

    geo_col = detect_geo_code_col(gdf)
    # normalize geo column and dept codes for merge
    gdf[geo_col] = gdf[geo_col].astype(str).str.strip()

    # Prepare data aggregation by departement
    map_df = q.groupby('departement_de_domicile', dropna=True)['valeur'].mean().reset_index().rename(columns={'departement_de_domicile':geo_col})

    # Ensure formatting: match leading zeros if needed
    # try zfill on numeric-looking codes
    def normalize_code(s):
        s = str(s).strip()
        if s.isdigit() and len(s) == 1:
            return s.zfill(2)
        return s

    gdf[geo_col] = gdf[geo_col].apply(lambda x: normalize_code(x))
    map_df[geo_col] = map_df[geo_col].apply(lambda x: normalize_code(x))

    merged = gdf.merge(map_df, on=geo_col, how='left')

    # Create animated choropleth with Plotly (auto play from min year to max year)
    # Build aggregated dataframe for all years (respecting selected sexe filter)
    map_df_all = df.copy()
    if sexe != 'Tous':
        map_df_all = map_df_all[map_df_all['sexe'] == sexe]

    # aggregate mean valeur by year and department
    map_df_all = map_df_all.groupby(['annee_de_deces', 'departement_de_domicile'], dropna=True)['valeur'].mean().reset_index()
    # rename to match geo column for plotting
    map_df_all = map_df_all.rename(columns={'departement_de_domicile': geo_col})

    # normalize codes same way as geo
    map_df_all[geo_col] = map_df_all[geo_col].astype(str).str.strip().apply(lambda x: normalize_code(x))

    try:
        # use the GeoJSON representation from geopandas
        geojson = gdf.__geo_interface__
        # ensure animation frame is string for correct ordering in plotly
        map_df_all['annee_de_deces'] = map_df_all['annee_de_deces'].astype(str)

        fig_map = px.choropleth(
            map_df_all,
            geojson=geojson,
            locations=geo_col,
            color='valeur',
            featureidkey=f"properties.{geo_col}",
            animation_frame='annee_de_deces',
            projection='mercator',
            color_continuous_scale='YlOrRd',
            labels={'valeur': 'Taux moyen'},
            height=700
        )

        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

        # add custom play/pause buttons with user-selected speed
        fig_map.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': int(speed_ms), 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 300}}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 70},
                'showactive': False,
                'x': 0.1,
                'y': 0
            }]
        )

        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la création de la carte animée: {e}")


if __name__ == '__main__':
    main()
