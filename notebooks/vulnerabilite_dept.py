import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json


st.set_page_config(page_title="Vulnérabilité par département", layout="wide")


def try_read_csv(path_or_buffer):
    # Try semicolon with decimal comma first (common in French datasets)
    try:
        df = pd.read_csv(path_or_buffer, sep=';', decimal=',', engine='python')
        return df
    except Exception:
        pass
    # Try comma, dot decimal
    try:
        df = pd.read_csv(path_or_buffer, sep=',', decimal='.', engine='python')
        return df
    except Exception:
        pass
    # Last resort: let pandas infer
    return pd.read_csv(path_or_buffer, engine='python')


def to_numeric_series(s):
    # Replace common thousands and decimal separators then convert
    if s.dtype == object:
        s = s.str.replace(' ', '')
        s = s.str.replace('\u00A0', '')
        s = s.str.replace(',', '.')
    return pd.to_numeric(s, errors='coerce')


st.title("Table de vulnérabilité par département")
st.markdown(
    "Cet outil calcule un indice composite de vulnérabilité en combinant des métriques (ex: CSP, pauvreté, mortalité).\n"
    "Vous pouvez charger votre CSV ou utiliser un exemple du dépôt."
)

SAMPLE_CSVS = [
    'notebooks/Category_Profession.csv',
    'src/summary_low_by_dept.csv',
    'src/summary_missing_by_dept.csv',
    'src/dim_geo_departement.csv',
]

available_samples = [p for p in SAMPLE_CSVS if os.path.exists(os.path.join(os.getcwd(), p))]

with st.sidebar:
    st.header('Source de données')
    source = st.radio('Charger depuis', ['Exemple fourni', 'Uploader un CSV'], index=0)
    chosen_sample = None
    uploaded_file = None
    if source == 'Exemple fourni':
        if available_samples:
            chosen_sample = st.selectbox('Choisir un fichier exemple', available_samples)
        else:
            st.write('Aucun exemple trouvé dans le dépôt.')
    else:
        uploaded_file = st.file_uploader('Déposer un fichier CSV', type=['csv'])

    st.markdown('---')
    st.header('Colonnes / Indicateurs')
    show_preview = st.checkbox('Afficher un aperçu des données après chargement', value=True)


df = None
if source == 'Exemple fourni' and chosen_sample:
    path = os.path.join(os.getcwd(), chosen_sample)
    try:
        df = try_read_csv(path)
    except Exception as e:
        st.error(f"Impossible de lire le fichier exemple: {e}")
elif uploaded_file is not None:
    try:
        df = try_read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Impossible de lire le fichier uploadé: {e}")
else:
    st.info('Choisissez un fichier exemple ou uploadez un CSV pour commencer.')

if df is not None:
    st.write(f'Données chargées — {df.shape[0]} lignes × {df.shape[1]} colonnes')
    if show_preview:
        st.dataframe(df.head())

    cols = df.columns.tolist()
    with st.sidebar:
        dept_col = st.selectbox('Colonne département / identifiant', [c for c in cols if 'dept' in c.lower() or 'départ' in c.lower() or 'department' in c.lower()] + cols, index=0)
        st.markdown('Sélectionnez les 2–4 indicateurs à combiner (ex: pauvreté, mortalité, CSP)')
        indicator_cols = st.multiselect('Indicateurs (colonnes numériques)', cols, default=[c for c in cols if any(k in c.lower() for k in ['pauv', 'mort', 'csp', 'vuln', 'taux', 'rate', 'prop'])][:3])
        st.markdown('Options de calcul')
        weight_text = st.text_input('Poids (virgule séparés) correspondant aux indicateurs sélectionnés — laisser vide = poids égaux', '')
        invert_choices = st.multiselect('Inverser (si une colonne a une valeur plus élevée = moins vulnérable)', indicator_cols)
        n_top = st.number_input('Nombre de départements à afficher (top N)', min_value=1, max_value=500, value=10)
        compute = st.button('Calculer l’indice de vulnérabilité')

    if compute:
        if not indicator_cols:
            st.error('Sélectionnez au moins une colonne indicatrice pour calculer l’indice.')
        else:
            df_work = df.copy()
            # Ensure department column exists
            if dept_col not in df_work.columns:
                st.error('La colonne département n’a pas été trouvée dans le fichier.')
            else:
                # Prepare numeric indicators
                for c in indicator_cols:
                    df_work[c] = to_numeric_series(df_work[c])
                    # invert if requested
                    if c in invert_choices:
                        df_work[c] = -df_work[c]

                # Weights
                if weight_text.strip():
                    try:
                        weights = [float(w) for w in weight_text.replace(',', '.').split(',')]
                        if len(weights) != len(indicator_cols):
                            st.warning('Le nombre de poids ne correspond pas au nombre d’indicateurs — poids égaux appliqués.')
                            weights = None
                    except Exception:
                        st.warning('Impossible d’interpréter les poids — poids égaux appliqués.')
                        weights = None
                else:
                    weights = None

                # Compute z-scores per indicator, ignoring NaNs
                zcols = []
                for c in indicator_cols:
                    zname = f'z__{c}'
                    series = df_work[c]
                    mean = series.mean(skipna=True)
                    std = series.std(skipna=True)
                    if pd.isna(std) or std == 0:
                        df_work[zname] = 0.0
                    else:
                        df_work[zname] = (series - mean) / std
                    zcols.append(zname)

                if weights is None:
                    weights = [1.0] * len(zcols)

                # Compute weighted composite (higher = more vulnerable)
                w = np.array(weights, dtype=float)
                w = w / w.sum()
                df_work['vulnerability_index'] = df_work[zcols].fillna(0).values.dot(w)
                df_work['vulnerability_rank'] = df_work['vulnerability_index'].rank(ascending=False, method='min')

                # Build result table grouped by department (first occurrence)
                result = df_work[[dept_col] + indicator_cols + ['vulnerability_index', 'vulnerability_rank']].copy()
                # If multiple rows per department, aggregate by mean
                grouped = result.groupby(dept_col).agg({**{c: 'mean' for c in indicator_cols}, 'vulnerability_index': 'mean', 'vulnerability_rank': 'min'})
                grouped = grouped.sort_values('vulnerability_index', ascending=False)

                st.subheader('Tableau de vulnérabilité par département')
                st.dataframe(grouped.reset_index().head(200))

                st.subheader(f'Top {n_top} départements les plus vulnérables')
                topn = grouped.reset_index().head(n_top)
                fig = px.bar(topn, x=dept_col, y='vulnerability_index', color='vulnerability_index', color_continuous_scale='Reds', title='Top départements par indice de vulnérabilité')
                st.plotly_chart(fig, use_container_width=True)

                st.subheader(f'Bottom {n_top} départements (les moins vulnérables)')
                bottomn = grouped.reset_index().tail(n_top)
                fig2 = px.bar(bottomn, x=dept_col, y='vulnerability_index', color='vulnerability_index', color_continuous_scale='Blues', title='Bottom départements par indice de vulnérabilité')
                st.plotly_chart(fig2, use_container_width=True)

                # Optional geojson choropleth
                geojson_path = None
                candidate_geo = os.path.join(os.getcwd(), 'dev2', 'departements.geojson')
                if os.path.exists(candidate_geo):
                    geojson_path = candidate_geo

                st.markdown('---')
                st.write('Export')
                csv = grouped.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button('Télécharger le tableau (CSV)', data=csv, file_name='vulnerabilite_dept.csv', mime='text/csv')

                if geojson_path:
                    try:
                        with open(geojson_path, 'r', encoding='utf-8') as f:
                            gj = json.load(f)
                        # Attempt to find an ID property in geojson features
                        # Ask user to provide the property name if necessary
                        st.subheader('Carte choroplèthe (optionnelle)')
                        geo_prop = st.text_input('Nom de la propriété dans le GeoJSON correspondant au département (laisser vide pour tenter automatiquement)', '')
                        # Attempt to auto-detect a likely property name
                        if not geo_prop:
                            sample_props = gj['features'][0]['properties'].keys()
                            candidates = [p for p in sample_props if any(k in p.lower() for k in ['nom', 'name', 'dep', 'code'])]
                            geo_prop = candidates[0] if candidates else list(sample_props)[0]

                        # Prepare dataframe for mapping
                        map_df = grouped.reset_index().copy()
                        map_df[dept_col] = map_df[dept_col].astype(str)

                        figmap = px.choropleth_mapbox(map_df, geojson=gj, locations=dept_col, featureidkey=f'properties.{geo_prop}', color='vulnerability_index', mapbox_style='carto-positron', zoom=4.8, center={'lat':46.5,'lon':2.5}, color_continuous_scale='Reds', opacity=0.6)
                        st.plotly_chart(figmap, use_container_width=True)
                    except Exception as e:
                        st.warning(f'Impossible de générer la carte: {e}')

                # Mark tasks completed
                st.success('Calcul terminé — vous pouvez télécharger le tableau ou ajuster les paramètres.')

    st.markdown('---')
    st.write('Notes:')
    st.write('- Si vos nombres utilisent la virgule comme séparateur décimal, l’import tentera de le gérer automatiquement.')
    st.write('- Si vos colonnes ont des sens inverses (plus grand = meilleur), cochez-les dans la liste "Inverser" pour les renverser avant le calcul.')

else:
    st.info('Aucun jeu de données chargé — commencez par choisir un fichier exemple ou uploader votre CSV.')


if __name__ == '__main__':
    pass
