import os
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Optional

# import centralized data loaders
from data_loaders import (
    read_csv_flexible,
    load_rpps,
    load_departements,
    load_dim_fact,
    load_quentin_tables,
)

# Unified Streamlit app combining: app_rpps.py, vulnerabilite_dept.py, app_dev1.py, streamlit_viz.py
# Goal: simple sidebar to pick a tool, robust loaders, concise summaries and easy exports.

BASE = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Santé & Territoires — Exploration", layout="wide")


# data loaders are provided by `src/data_loaders.py` and imported above


def main():
    st.title("Santé & Territoires — Dashboard")

    st.sidebar.header("Outils")
    mode = st.sidebar.radio("Choisir une vue / outil", [
        'RPPS — Densité',
        'Vulnérabilité ',
        'Charge pathologique',
        'Exploration',
    ])

    if mode == 'RPPS — Densité':
        show_rpps()
    elif mode == 'Vulnérabilité par département':
        show_vulnerabilite()
    elif mode == 'Charge pathologique':
        show_dev1()
    elif mode == 'Exploration':
        show_quentin()


def show_rpps():
    st.header('Densité des médecins par spécialité et département (RPPS)')
    # show a themed loading animation while reading files
    placeholder = st.empty()
    try:
        display_loading_animation(placeholder, variant='medecin')
        df = load_rpps()
        gdf_dept = load_departements()
    finally:
        placeholder.empty()

    if df.empty:
        st.warning('Données RPPS introuvables — placez `src/rpps_stream/rpps_long_clean.csv` ou équivalent.')
        return

    specialites = sorted(df['specialite'].dropna().unique().tolist())
    # allow selecting multiple specialties and provide an option to select all
    all_label = '(Toutes les spécialités)'
    options = [all_label] + specialites
    selected_specialites = st.multiselect('Choisir une ou plusieurs spécialités', options=options, default=[all_label])

    # interpret selection
    if not selected_specialites or all_label in selected_specialites:
        df_filtered = df.copy()
    else:
        df_filtered = df[df['specialite'].isin(selected_specialites)].copy()
    df_densite = df_filtered.groupby('dept_code_clean', as_index=False)['densite'].mean().sort_values('dept_code_clean')
    if df_densite.empty:
        st.info('Aucune donnée pour cette spécialité.')
        return

    # identifier colonne densité et colonne code département dans df_densite
    dens_col = next((c for c in df_densite.columns if 'dens' in c.lower()), None)
    dept_code_col = next((c for c in df_densite.columns if 'dept' in c.lower()), None)
    if dens_col is None or dept_code_col is None:
        st.error('Structure inattendue des données RPPS (colonne densité ou code département manquante).')
        return

    # Map names from gdf (si disponible)
    name_map = {}
    if not gdf_dept.empty and 'code' in gdf_dept.columns and 'nom' in gdf_dept.columns:
        name_map = gdf_dept.set_index('code')['nom'].to_dict()
    df_densite['dept_name'] = df_densite[dept_code_col].map(name_map).fillna(df_densite[dept_code_col])

    # ----- Résumé et KPI sous le titre -----
    st.markdown('**Résumé**')
    n_with = int(df_densite[dens_col].dropna().shape[0])
    n_total = int(len(gdf_dept)) if not gdf_dept.empty else int(df_densite.shape[0])
    n_missing = max(0, n_total - n_with)
    dens_min = float(df_densite[dens_col].min(skipna=True)) if df_densite[dens_col].notna().any() else None
    dens_max = float(df_densite[dens_col].max(skipna=True)) if df_densite[dens_col].notna().any() else None
    dens_mean = float(df_densite[dens_col].mean(skipna=True)) if df_densite[dens_col].notna().any() else None

    st.write('Utilisez la sélection pour comparer plusieurs spécialités et identifiez rapidement les départements sous-dotés.')
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric('Départements avec données', f'{n_with}')
    with k2:
        st.metric('Densité moyenne', f"{dens_mean:.2f}" if dens_mean is not None else 'N/A')
    with k3:
        st.metric('Densité min', f"{dens_min:.2f}" if dens_min is not None else 'N/A')
    with k4:
        st.metric('Densité max', f"{dens_max:.2f}" if dens_max is not None else 'N/A')

    st.subheader('Tableau résumé')
    # include both code and human-readable name
    if 'dept_name' not in df_densite.columns:
        df_densite['dept_name'] = df_densite[dept_code_col].fillna(df_densite[dept_code_col])
    display_df = df_densite[[dept_code_col, 'dept_name', dens_col]].copy().rename(columns={dept_code_col: 'Code', 'dept_name': 'Département', dens_col: 'Densité'})
    # trier automatiquement par densité croissante, NaN en fin
    display_df = display_df.sort_values('Densité', ascending=True, na_position='last')
    st.dataframe(display_df.reset_index(drop=True), width='stretch')

    # Quick summary stats
    st.markdown('**Résumé rapide**')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Départements (avec données)', int(display_df['Densité'].count()))
    with col2:
        st.metric('Densité moyenne', f"{display_df['Densité'].mean():.2f}")
    with col3:
        st.metric('Densité médiane', f"{display_df['Densité'].median():.2f}")

    st.subheader('Histogramme des densités')
    # Exclure les départements numériques entre 96 et 970
    def code_in_exclude_range(code_str):
        try:
            if isinstance(code_str, str) and code_str.isdigit():
                n = int(code_str)
                return 96 <= n <= 970
        except Exception:
            pass
        return False

    # Exclure également les lignes sommaires comme 'Total' / 'Total général'
    mask_total = display_df['Département'].astype(str).str.contains('total', case=False, na=False) | display_df['Code'].astype(str).str.contains('total', case=False, na=False)
    hist_df = display_df[~display_df['Code'].apply(code_in_exclude_range) & ~mask_total].copy()
    fig = px.bar(hist_df.sort_values('Densité', ascending=True), x='Département', y='Densité')
    st.plotly_chart(fig, width='stretch')

    # Map
    st.subheader('Carte choroplèthe (par département)')
    if gdf_dept.empty:
        st.info('GeoJSON des départements introuvable (utilisé pour la carte).')
    else:
        # merge en cherchant la colonne code de département
        merged = gdf_dept.merge(df_densite, left_on='code', right_on=dept_code_col, how='left')
        dens_col_merged = next((c for c in merged.columns if 'dens' in c.lower()), None)
        if dens_col_merged is None:
            st.warning('Aucune colonne de densité après jointure — la carte ne peut pas être colorée.')
        else:
            vmin = merged[dens_col_merged].min(skipna=True)
            vmax = merged[dens_col_merged].max(skipna=True)
            merged['fill_color'] = merged[dens_col_merged].apply(lambda v: dens_to_color(v, vmin, vmax))
            # ensure a human-readable name is present in properties for popups/tooltips
            if 'nom' not in merged.columns:
                merged['nom'] = merged.get('dept_name', merged[dept_code_col]).astype(str)
            else:
                merged['nom'] = merged['nom'].fillna(merged.get('dept_name', merged[dept_code_col]).astype(str))
            # create a uniform density property for tooltips
            merged['dens_val'] = pd.to_numeric(merged[dens_col_merged], errors='coerce')

            # Prepare a simplified geometry to speed up rendering
            try:
                merged_s = merged.copy()
                # project to web mercator for meter-based simplify, fallback to degree simplify
                try:
                    merged_proj = merged_s.to_crs(epsg=3857)
                    merged_proj['geometry'] = merged_proj['geometry'].simplify(tolerance=2000)
                    merged_s = merged_proj.to_crs(epsg=4326)
                except Exception:
                    merged_s['geometry'] = merged_s['geometry'].simplify(tolerance=0.005)

                # build one GeoJson layer (faster than adding per-feature objects)
                try:
                    import folium
                    from streamlit_folium import st_folium

                    centroid = merged_s.geometry.centroid
                    avg_lat = float(centroid.y.mean()) if not centroid.empty else 46.5
                    avg_lon = float(centroid.x.mean()) if not centroid.empty else 2.5
                    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6, tiles='cartodbpositron')

                    def style_function(feature):
                        props = feature.get('properties', {})
                        color = props.get('fill_color', [200,200,200,120])
                        r,g,b,a = color[:4]
                        return {
                            'fillColor': f'rgba({r},{g},{b},{a/255:.2f})',
                            'color': 'black',
                            'weight': 0.5,
                            'fillOpacity': a/255 if a else 0.6,
                        }

                    gj = merged_s.__geo_interface__
                    # ensure properties have nom and dens_val formatted
                    for feat in gj.get('features', []):
                        props = feat.get('properties', {})
                        name = props.get('nom') or props.get('dept_name') or props.get(dept_code_col)
                        dens = props.get('dens_val') if props.get('dens_val') is not None else props.get(dens_col_merged)
                        props['nom'] = name
                        props['dens_val'] = f"{float(dens):.2f}" if dens not in (None, '') and pd.notna(dens) else 'N/A'
                        feat['properties'] = props

                    geojson_layer = folium.GeoJson(gj, style_function=style_function, tooltip=folium.GeoJsonTooltip(fields=['nom','dens_val'], aliases=['Département','Densité'], localize=True))
                    geojson_layer.add_to(m)
                    st_folium(m, width='100%', height=600)
                except Exception:
                    # fallback to pydeck with simplified geometries
                    geojson = merged_s.__geo_interface__
                    view_state = pdk.ViewState(latitude=46.5, longitude=2.5, zoom=4.5)
                    layer = pdk.Layer('GeoJsonLayer', data=geojson, stroked=False, filled=True, get_fill_color='properties.fill_color', pickable=True, auto_highlight=True)
                    tooltip = {"text": "Département {properties.nom} ({properties.code})\nDensité: {properties.dens_val}"}
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
            except Exception:
                st.warning('Impossible de générer la carte interactive.')

    # Downloads
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button('Télécharger le tableau (CSV)', data=csv, file_name='densite_par_departement.csv', mime='text/csv')


def dens_to_color(v, vmin, vmax):
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


def display_loading_animation(place: Optional[object], variant: str = 'ambulance'):
        """Render a small HTML/CSS animation into the provided placeholder (st.empty()).

        Supported variants: 'ambulance', 'medecin', 'brancard'.
        The placeholder should be cleared by calling `place.empty()` after loading completes.
        """
        if place is None:
                return
        # simple emoji/CSS animation fallback (no external assets)
        if variant == 'medecin':
                emoji = '🩺'
                text = 'Chargement des données — médecin en consultation'
        elif variant == 'brancard':
                emoji = '🛌'
                text = 'Chargement — brancard en route'
        else:
                emoji = '🚑'
                text = 'Chargement des données — ambulance en approche'

        html = f"""
        <div style="display:flex;align-items:center;gap:16px;padding:8px 0;">
            <style>
                .slide-wrap{{position:relative;overflow:hidden;height:80px}}
                .vehicle{{position:absolute;left:-20%;top:8px;font-size:56px;animation:drive 2.2s linear infinite}}
                .pulse{{font-size:18px;color:#333;font-weight:600;}}
                @keyframes drive{{0%{{left:-30%;transform:rotate(0deg)}}50%{{left:55%;transform:rotate(-2deg)}}100%{{left:110%;transform:rotate(0deg)}}}}
                .dot{{width:10px;height:10px;background:#e74c3c;border-radius:50%;display:inline-block;margin-left:8px;animation:beat 1s infinite}}
                @keyframes beat{{0%{{transform:scale(1)}}50%{{transform:scale(1.5)}}100%{{transform:scale(1)}}}}
            </style>
            <div style="flex:0 0 320px;">
                <div class="slide-wrap">
                    <div class="vehicle">{emoji}</div>
                </div>
            </div>
            <div style="flex:1;">
                <div class="pulse">{text} <span class="dot"></span></div>
                <div style="color:#666;margin-top:6px;font-size:13px">Patientez — affichage dès que les données sont prêtes.</div>
            </div>
        </div>
        """
        try:
                place.markdown(html, unsafe_allow_html=True)
        except Exception:
                try:
                        # fallback: simple message
                        place.text(f"{text} ...")
                except Exception:
                        pass


def show_vulnerabilite():
    st.header('Vulnérabilité par département')
    st.write('Calcul d’un indice composite à partir d’un CSV (exemples inclus).')
    # allow upload or sample
    sample_paths = [
        'notebooks/Category_Profession.csv',
        'src/summary_low_by_dept.csv',
        'src/summary_missing_by_dept.csv',
        'src/dim_geo_departement.csv',
    ]
    available = [p for p in sample_paths if os.path.exists(os.path.join(os.getcwd(), p))]

    with st.sidebar:
        source = st.radio('Source de données', ['Exemple fourni', 'Uploader un CSV'], index=0)
        chosen = None
        upload = None
        if source == 'Exemple fourni' and available:
            chosen = st.selectbox('Choisir un exemple', available)
        elif source == 'Uploader un CSV':
            upload = st.file_uploader('Déposer un CSV', type=['csv'])

    df = None
    if chosen:
        df = read_csv_flexible(os.path.join(os.getcwd(), chosen))
    if upload is not None:
        df = read_csv_flexible(upload)

    if df is None:
        st.info('Aucun jeu de données chargé — utilisez la sidebar pour uploader ou choisir un exemple.')
        return

    st.write(f'Données chargées — {df.shape[0]} lignes × {df.shape[1]} colonnes')
    if st.checkbox('Afficher aperçu', value=True):
        st.dataframe(df.head())

    cols = df.columns.tolist()
    with st.sidebar:
        dept_col = st.selectbox('Colonne département', [c for c in cols if 'dept' in c.lower() or 'départ' in c.lower() or 'department' in c.lower()] + cols)
        indicator_cols = st.multiselect('Indicateurs numériques (2–4)', cols, default=[c for c in cols if any(k in c.lower() for k in ['pauv', 'mort', 'csp', 'vuln', 'taux', 'rate', 'prop'])][:3])
        weight_text = st.text_input('Poids (virgule séparés)', '')
        invert_choices = st.multiselect('Inverser colonnes (plus grand = moins vulnérable)', indicator_cols)
        n_top = st.number_input('Top N', min_value=1, max_value=500, value=10)
        compute = st.button('Calculer indice')

    if compute:
        if not indicator_cols:
            st.error('Sélectionnez au moins un indicateur.')
            return
        df_work = df.copy()
        if dept_col not in df_work.columns:
            st.error('Colonne département introuvable.')
            return
        # numeric conversion
        for c in indicator_cols:
            if df_work[c].dtype == object:
                df_work[c] = df_work[c].astype(str).str.replace(' ', '').str.replace('\u00A0', '').str.replace(',', '.').replace('', np.nan)
            df_work[c] = pd.to_numeric(df_work[c], errors='coerce')
            if c in invert_choices:
                df_work[c] = -df_work[c]

        # weights
        if weight_text.strip():
            try:
                weights = [float(w.replace(',', '.')) for w in weight_text.split(',')]
                if len(weights) != len(indicator_cols):
                    st.warning('Nombre de poids différent ; poids égaux utilisés.')
                    weights = None
            except Exception:
                st.warning('Impossible d’interpréter les poids ; poids égaux utilisés.')
                weights = None
        else:
            weights = None

        zcols = []
        for c in indicator_cols:
            zname = f'z__{c}'
            s = df_work[c]
            mean = s.mean(skipna=True)
            std = s.std(skipna=True)
            if pd.isna(std) or std == 0:
                df_work[zname] = 0.0
            else:
                df_work[zname] = (s - mean) / std
            zcols.append(zname)

        if weights is None:
            weights = [1.0] * len(zcols)
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        df_work['vulnerability_index'] = df_work[zcols].fillna(0).values.dot(w)
        df_work['vulnerability_rank'] = df_work['vulnerability_index'].rank(ascending=False, method='min')

        grouped = df_work.groupby(dept_col).agg({**{c: 'mean' for c in indicator_cols}, 'vulnerability_index': 'mean', 'vulnerability_rank': 'min'})
        grouped = grouped.sort_values('vulnerability_index', ascending=False)

        st.subheader('Tableau par département')
        st.dataframe(grouped.reset_index().head(200))

        st.subheader(f'Top {n_top} départements — vulnérabilité élevée')
        topn = grouped.reset_index().head(n_top)
        fig = px.bar(topn, x=dept_col, y='vulnerability_index', color='vulnerability_index', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

        # export
        csv = grouped.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button('Télécharger le tableau (CSV)', data=csv, file_name='vulnerabilite_dept.csv', mime='text/csv')


def show_dev1():
    st.header('Charge pathologique par département et spécialité')
    placeholder = st.empty()
    try:
        display_loading_animation(placeholder, variant='brancard')
        dim, fact = load_dim_fact()
    finally:
        placeholder.empty()
    if dim is None or fact is None:
        st.warning('Fichiers `dim_geo_departement.csv` et `fact_dep_specialite_patho.csv` introuvables. Placez-les dans `notebooks` ou `src/`.')
        return

    # harmonize codes
    if fact['geo_code'].dtype.kind in 'ifu':
        fact['geo_code'] = fact['geo_code'].astype(int).astype(str).str.zfill(2)
    else:
        fact['geo_code'] = fact['geo_code'].astype(str).str.zfill(2)
    dim_sub = dim.rename(columns={'geo_libelle': 'departement'})
    if 'geo_code' not in dim_sub.columns:
        st.error("Fichier `dim_geo_departement.csv` sans colonne attendue 'geo_code'.")
        return
    df = fact.merge(dim_sub[['geo_code', 'departement', 'code_region']], on='geo_code', how='left')
    df = df.dropna(subset=['departement'])

    # filters
    all_departements = sorted(df['departement'].dropna().unique().tolist())
    all_specialites = sorted(df['specialite'].dropna().unique().tolist())
    all_regions = sorted(df['code_region'].dropna().unique().tolist())

    st.sidebar.header('Filtres')
    departement_sel = st.sidebar.multiselect('Départements', options=all_departements, default=all_departements)
    specialite_sel = st.sidebar.multiselect('Spécialités', options=all_specialites, default=all_specialites)
    region_sel = st.sidebar.multiselect('Régions', options=all_regions, default=all_regions)
    top_n = st.sidebar.slider('Top N', min_value=5, max_value=50, value=15)

    df_filtre = df.copy()
    if departement_sel:
        df_filtre = df_filtre[df_filtre['departement'].isin(departement_sel)]
    if specialite_sel:
        df_filtre = df_filtre[df_filtre['specialite'].isin(specialite_sel)]
    if region_sel:
        df_filtre = df_filtre[df_filtre['code_region'].isin(region_sel)]

    st.write(f'Lignes après filtre : {len(df_filtre):,}'.replace(',', ' '))

    # --- Résumé & KPI (nouveau)
    # Make sure charge_pathologique is numeric
    if 'charge_pathologique' in df_filtre.columns:
        df_filtre['charge_pathologique'] = pd.to_numeric(df_filtre['charge_pathologique'], errors='coerce')
    total_charge = float(df_filtre['charge_pathologique'].sum(skipna=True)) if 'charge_pathologique' in df_filtre.columns else None
    n_departements = df_filtre['departement'].nunique() if 'departement' in df_filtre.columns else 0
    avg_per_dept = (total_charge / n_departements) if total_charge is not None and n_departements else None
    # top department
    top_dept = None
    if 'charge_pathologique' in df_filtre.columns and 'departement' in df_filtre.columns:
        tmp = df_filtre.groupby('departement', as_index=False)['charge_pathologique'].sum().sort_values('charge_pathologique', ascending=False)
        if not tmp.empty:
            top_dept = (tmp.iloc[0]['departement'], float(tmp.iloc[0]['charge_pathologique']))
    # top specialty
    top_spec = None
    if 'charge_pathologique' in df_filtre.columns and 'specialite' in df_filtre.columns:
        tmp2 = df_filtre.groupby('specialite', as_index=False)['charge_pathologique'].sum().sort_values('charge_pathologique', ascending=False)
        if not tmp2.empty:
            top_spec = (tmp2.iloc[0]['specialite'], float(tmp2.iloc[0]['charge_pathologique']))

    st.markdown('**Résumé rapide (à retenir)**')
    st.write('Ces graphiques montrent la charge pathologique agrégée par département et par spécialité selon les filtres appliqués. Utilisez les filtres à gauche pour explorer par région, département ou spécialité.')
    st.write('Principaux KPI ci‑dessous : Totaux, moyenne par département, et les top contributeurs.')

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if total_charge is not None:
            st.metric('Charge totale', f"{total_charge:,.0f}".replace(',', ' '))
        else:
            st.metric('Charge totale', 'N/A')
    with k2:
        if avg_per_dept is not None:
            st.metric('Moyenne par département', f"{avg_per_dept:,.1f}".replace(',', ' '))
        else:
            st.metric('Moyenne par département', 'N/A')
    with k3:
        if top_dept:
            st.metric('Département le plus chargé', f"{top_dept[0]} ({top_dept[1]:,.0f})".replace(',', ' '))
        else:
            st.metric('Département le plus chargé', 'N/A')
    with k4:
        if top_spec:
            st.metric('Spécialité la plus chargée', f"{top_spec[0]} ({top_spec[1]:,.0f})".replace(',', ' '))
        else:
            st.metric('Spécialité la plus chargée', 'N/A')


    agg_dep = df_filtre.groupby('departement', as_index=False)['charge_pathologique'].sum().sort_values('charge_pathologique', ascending=False)
    agg_spec = df_filtre.groupby('specialite', as_index=False)['charge_pathologique'].sum().sort_values('charge_pathologique', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Charge par département')
        st.dataframe(agg_dep.head(top_n).reset_index(drop=True), width='stretch')
        if not agg_dep.empty:
            st.bar_chart(agg_dep.head(top_n).set_index('departement')['charge_pathologique'])
    with col2:
        st.subheader('Charge par spécialité')
        st.dataframe(agg_spec.head(top_n).reset_index(drop=True), width='stretch')
        if not agg_spec.empty:
            st.bar_chart(agg_spec.head(top_n).set_index('specialite')['charge_pathologique'])

    st.markdown('### Télécharger données filtrées')
    csv_bytes = df_filtre.to_csv(index=False).encode('utf-8')
    st.download_button('Télécharger (CSV)', data=csv_bytes, file_name='filtered_data.csv', mime='text/csv')


def show_quentin():
    st.header('Exploration — Offres & Pathologies')
    placeholder = st.empty()
    try:
        display_loading_animation(placeholder, variant='ambulance')
        res = load_quentin_tables()
    finally:
        placeholder.empty()
    df_offre = res.get('offre')
    df_med = res.get('med')
    df_dept = res.get('dept')
    gdf = res.get('gdf')

    if df_offre is None and df_med is None:
        st.info('Aucune table d’offres / médecins trouvée dans `notebooks/Quentin`.')
    else:
        famille_choices = ['(Tous)']
        if df_offre is not None and 'famille_pathologie' in df_offre.columns:
            famille_choices += sorted(df_offre['famille_pathologie'].dropna().unique().tolist())
        famille = st.sidebar.selectbox('Famille pathologie', famille_choices)

        spec_choices = ['(Tous)']
        if df_offre is not None and 'specialite' in df_offre.columns:
            spec_choices += sorted(df_offre['specialite'].dropna().unique().tolist())
        specialite = st.sidebar.selectbox('Spécialité', spec_choices)

        st.subheader('Carte : points d’offre (médecins / structures)')
        map_df = None
        if df_offre is not None and {'longitude', 'latitude'}.issubset(df_offre.columns):
            map_df = df_offre.copy()
            if famille != '(Tous)':
                map_df = map_df[map_df['famille_pathologie'] == famille]
            if specialite != '(Tous)':
                map_df = map_df[map_df['specialite'] == specialite]
            map_df = map_df.dropna(subset=['longitude', 'latitude'])

        if (map_df is None or map_df.empty) and df_med is not None and {'longitude', 'latitude'}.issubset(df_med.columns):
            map_df = df_med.copy()

        if map_df is None or map_df.empty:
            st.info('Aucune donnée de points géographiques trouvée pour ces filtres.')
        else:
            midpoint = (map_df['latitude'].mean(), map_df['longitude'].mean())
            st.write(f'Points affichés : {len(map_df)}')
            layer = pdk.Layer('ScatterplotLayer', data=map_df, get_position='[longitude, latitude]', get_radius=100, get_fill_color='[200, 30, 0, 160]', pickable=True)
            view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=6)
            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={'text': '{specialite}\n{famille_pathologie}'})
            st.pydeck_chart(r)

        st.markdown('---')
        if df_dept is None or gdf is None:
            st.info('Fichier départemental ou GeoJSON introuvable pour choroplèthe.')
        else:
            st.subheader('Choroplèthe par département')
            # attempt to find code column
            if 'ZONE_0' in df_dept.columns:
                df_dept['ZONE_0'] = df_dept['ZONE_0'].astype(str).str.zfill(2)
                dept_code_col = find_dept_code_col(gdf, df_dept['ZONE_0'])
                if dept_code_col is None:
                    st.warning('Impossible d’identifier colonne code département dans le GeoJSON.')
                else:
                    gdf[dept_code_col] = gdf[dept_code_col].astype(str).str.zfill(2)
                    df_dep_agg = df_dept.groupby('ZONE_0', as_index=False).agg({'ind_freq_sum': 'sum'})
                    merged = gdf.merge(df_dep_agg, left_on=dept_code_col, right_on='ZONE_0', how='left')
                    merged['ind_freq_sum'] = merged['ind_freq_sum'].fillna(0)
                    poly_layer = choropleth_layer_from_gdf(merged, 'ind_freq_sum')
                    if poly_layer is not None:
                        centroid = merged.geometry.centroid
                        avg_lat = centroid.y.mean()
                        avg_lon = centroid.x.mean()
                        view_state = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=5)
                        deck = pdk.Deck(layers=[poly_layer], initial_view_state=view_state)
                        st.pydeck_chart(deck)
                    st.table(merged[['ZONE_0', 'ind_freq_sum']].sort_values('ind_freq_sum', ascending=False).head(10).set_index('ZONE_0'))


def find_dept_code_col(gdf, df_codes):
    if gdf is None or df_codes is None:
        return None
    candidates = []
    for c in gdf.columns:
        if gdf[c].dtype == object or pd.api.types.is_integer_dtype(gdf[c]):
            try:
                common = set(gdf[c].astype(str).str.zfill(2).values) & set(df_codes.astype(str).str.zfill(2).values)
                if len(common) > 0:
                    candidates.append((c, len(common)))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def choropleth_layer_from_gdf(gdf, value_col, value_min=None, value_max=None):
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        try:
            coords = []
            if geom.geom_type == 'MultiPolygon':
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
    vals = [f['value'] for f in features]
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
        layer_data.append({"polygon": f['coordinates'], "value": f['value'], "color": color_for(f['value'])})
    return pdk.Layer('PolygonLayer', layer_data, get_polygon='polygon', get_fill_color='color', pickable=True, auto_highlight=True, stroked=False, extruded=False)


if __name__ == '__main__':
    main()
