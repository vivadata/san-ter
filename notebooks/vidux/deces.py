import streamlit as st
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import traceback
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    px = None
    _HAS_PLOTLY = False
from datetime import datetime
from pathlib import Path
import unicodedata
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="📊 Analyse des Décès France 2018-2023",
    page_icon="💀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .header-title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Charger les données
@st.cache_data
def load_data():
    fname = 'data deces 2018-2023 - Données.csv'
    candidates = [
        Path(fname),
        Path('notebooks') / fname,
        Path(__file__).resolve().parent.parent / fname,
        Path(__file__).resolve().parent / '..' / fname,
    ]
    for p in candidates:
        try:
            if p.exists():
                return pd.read_csv(p)
        except Exception:
            import streamlit as st
            import pandas as pd
            import plotly.express as px
            from pathlib import Path
            import unicodedata

            st.set_page_config(page_title="📊 Analyse des Décès France (2018-2023)", page_icon="💀", layout="wide")

            # Simple styling
            st.markdown("""
                <style>
                .header-title { text-align: center; color: #2c3e50; margin-bottom: 0.5rem; }
                </style>
            """, unsafe_allow_html=True)

            @st.cache_data
            def load_data():
                fname = 'data deces 2018-2023 - Données.csv'
                candidates = [Path(fname), Path('notebooks') / fname, Path(__file__).resolve().parent.parent / fname]
                for p in candidates:
                    try:
                        if p.exists():
                            return pd.read_csv(p)
                    except Exception:
                        continue
                return pd.DataFrame()


            df = load_data()

            # Debug helpers: show basic status so a blank page reveals what's wrong
            try:
                st.markdown('**Debug:**')
                st.write('Dataframe empty:', df.empty)
                st.write('Dataframe shape:', getattr(df, 'shape', None))
                st.write('First columns (up to 50):', list(df.columns)[:50])
            except Exception:
                # If Streamlit/display fails early, avoid raising
                pass


            def _norm(s):
                return ''.join(ch for ch in unicodedata.normalize('NFKD', str(s)).lower() if ord(ch) < 128 and (ch.isalnum() or ch.isspace()))


            def find_col(df, candidates):
                cols = list(df.columns)
                norm_map = { _norm(c): c for c in cols }
                for cand in candidates:
                    nc = _norm(cand)
                    if nc in norm_map:
                        return norm_map[nc]
                return None


            # Detect columns
            deces_col = find_col(df, ['Nombre de décès', 'Nombre de deces', 'Nombre_deces', 'deces', 'nb_deces', 'nombre_deces'])
            year_col = find_col(df, ['Année', 'Annee', 'Year'])
            cause_col = find_col(df, ['Cause initiale de deces', 'cause_initiale_de_deces', 'cause', 'cause_initiale', 'Cause'])
            age_class_col = find_col(df, ['Grandes_classes_age', 'grandes_classes_age', 'classe_age', 'age_group', 'age_class'])
            dept_col = find_col(df, ['Département', 'Departement', 'dept', 'department'])
            dept_dom_col = find_col(df, ['departement_de_domicile', 'departement domicile', 'departement_domicile', 'dept_domicile'])

            has_deces = deces_col is not None

            # Header
            try:
                st.markdown("<h1 class='header-title'>📊 Analyse des Décés en France (2018-2023)</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align:center; color:gray;'>Étude démographique (2018-2023)</p>", unsafe_allow_html=True)
                st.divider()

            # Compute total deaths
            if has_deces and deces_col in df.columns:
                try:
                    total_deces = int(pd.to_numeric(df[deces_col], errors='coerce').sum(skipna=True))
                except Exception:
                    total_deces = int(len(df))
            else:
                total_deces = int(len(df))

            # Prepare population and valeur detection for per-100k calculation
            valeur_col = find_col(df, ['valeur', 'Valeur', 'valeur_deces', 'val']) if not df.empty else None
            pop_col = find_col(df, ['Population', 'population', 'pop', 'pop_total', 'population_totale']) if not df.empty else None

            # Compute deces per 100k
            deces_per_100k = None
            caption_note = None
            if not df.empty:
                try:
                    if valeur_col and valeur_col in df.columns:
                        sum_valeur = pd.to_numeric(df[valeur_col], errors='coerce').sum(skipna=True)
                    else:
                        sum_valeur = float(total_deces)

                    if pop_col and pop_col in df.columns:
                        total_pop = pd.to_numeric(df[pop_col], errors='coerce').sum(skipna=True)
                        if total_pop and total_pop > 0:
                            deces_per_100k = (sum_valeur / total_pop) * 100000
                        else:
                            caption_note = 'Population invalide — affichage de la somme colonne `valeur`'
                    else:
                        if valeur_col and valeur_col in df.columns:
                            deces_per_100k = pd.to_numeric(df[valeur_col], errors='coerce').mean(skipna=True)
                            caption_note = 'Aucune population trouvée — moyenne de la colonne `valeur` affichée'
                        else:
                            caption_note = 'Aucune colonne `valeur` ni population trouvée — affichage total brut'
                except Exception:
                    deces_per_100k = None

            # Compute top categories for KPIs
            def _compute_top(col):
                if col is None or col not in df.columns:
                    return None, None
                try:
                    if has_deces and deces_col in df.columns:
                        s = df.groupby(col)[deces_col].sum().sort_values(ascending=False)
                    else:
                        s = df[col].value_counts()
                    if s.empty:
                        return None, None
                    top = s.index[0]
                    val = int(s.iloc[0]) if hasattr(s.iloc[0], '__int__') else float(s.iloc[0])
                    return top, val
                except Exception:
                    return None, None

            cause_top, cause_val = _compute_top(cause_col)
            age_top, age_val = _compute_top(age_class_col)
            dept_top, dept_val = _compute_top(dept_col)
            dept_dom_top, dept_dom_val = _compute_top(dept_dom_col)

            # Display 4 KPIs
            st.markdown('### 🎯 Les 4 KPIs Clés')
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if deces_per_100k is not None:
                    st.metric(label='📈 Total Décès par 100k habitant', value=f"{deces_per_100k:,.1f}")
                else:
                    st.metric(label='📈 Total Décès par 100k habitant', value=f"{total_deces:,}")
                if caption_note:
                    st.caption(caption_note)

            with col2:
                if age_top:
                    st.metric(label="Classe d'âge la plus touchée", value=str(age_top), delta=f"{age_val:,}")
                else:
                    st.metric(label="Classe d'âge la plus touchée", value='N/A')

            with col3:
                if cause_top:
                    st.metric(label="Cause la plus fréquente", value=str(cause_top), delta=f"{cause_val:,}")
                else:
                    st.metric(label="Cause la plus fréquente", value='N/A')

            with col4:
                if dept_dom_top:
                    st.metric('Département domicile le plus touché', str(dept_dom_top), delta=f"{dept_dom_val:,}")
                else:
                    st.metric('Département domicile le plus touché', 'N/A')


            # ==================== GRAPHIQUE: Causes de décès (du plus grand au plus petit) ====================
            st.markdown('### 📋 Répartition par cause de décès — du plus grand au plus petit')
            if df.empty:
                st.info('Données absentes — impossible d’afficher le graphique des causes.')
            else:
                if cause_col is None or cause_col not in df.columns:
                    st.info('Colonne de cause introuvable dans le fichier — vérifiez les noms de colonnes.')
                else:
                    try:
                        if has_deces and deces_col in df.columns:
                            s = df.groupby(cause_col)[deces_col].sum().sort_values(ascending=False)
                            df_causes = s.reset_index().rename(columns={deces_col: 'count'})
                        else:
                            s = df[cause_col].value_counts()
                            df_causes = s.reset_index().rename(columns={'index': cause_col, cause_col: 'count'})

                        df_causes = df_causes.head(50)

                        fig = px.bar(df_causes, x='count', y=cause_col, orientation='h',
                                     labels={'count': 'Nombre (ou somme des décès)', cause_col: 'Cause'},
                                     title='Causes de décès — triées du plus fréquent au moins fréquent')
                        # Ensure largest bars appear at top
                        fig.update_layout(yaxis={'categoryorder':'array', 'categoryarray': df_causes[cause_col].tolist()[::-1]})
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Erreur lors du calcul/affichage du graphique: {e}")
                        except Exception:
                            st.error("Une erreur empêche l'affichage de la page. Voir le traceback ci-dessous.")
                            st.code(traceback.format_exc())
                        s = df[cause_col].value_counts()

                        df_causes = s.reset_index().rename(columns={'index': cause_col, cause_col: 'count'})
