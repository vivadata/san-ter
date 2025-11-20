import os
from io import StringIO
import streamlit as st
import pandas as pd

# -----------------------------
# Config de la page
# -----------------------------
st.set_page_config(
    page_title="Santé & Territoires - Dev 1",
    layout="wide"
)

st.title("📊 Charge pathologique par département et spécialité")
st.write("Vue simple construite à partir de `dim_geo_departement.csv` et `fact_dep_specialite_patho.csv`.")

# -----------------------------
# Chargement des données
# -----------------------------
# Résolution robuste des chemins des fichiers de données.
# L'app cherchera ces fichiers dans plusieurs répertoires candidats.
CANDIDATE_DIRS = [
    "notebooks",
    "src/rpps_stream",
    "data",
    ".",
]

def find_file(filename: str):
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None

DIM_PATH = find_file('dim_geo_departement.csv') or 'notebooks/dim_geo_departement.csv'
FACT_PATH = find_file('fact_dep_specialite_patho.csv') or 'notebooks/fact_dep_specialite_patho.csv'

@st.cache_data
def load_csv_safe(path):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data
def load_data():
    dim = load_csv_safe(DIM_PATH)
    fact = load_csv_safe(FACT_PATH)

    if dim is None or fact is None:
        return None

    # Harmoniser le code département :
    # - dim : strings '01', '02', ..., '971', '976'
    # - fact : entiers 1, 2, ..., 971, 976
    # Convertir en int si possible puis en str zfill(2)
    if fact["geo_code"].dtype.kind in "ifu":
        fact["geo_code"] = fact["geo_code"].astype(int).astype(str).str.zfill(2)
    else:
        fact["geo_code"] = fact["geo_code"].astype(str).str.zfill(2)

    # Jointure sur le code géographique
    dim_sub = dim.rename(columns={"geo_libelle": "departement"})
    if "geo_code" not in dim_sub.columns:
        # si le fichier dim a un code différent, essayer de trouver
        raise ValueError("Fichier dim_geo_departement.csv sans colonne 'geo_code' attendue.")
    df = fact.merge(
        dim_sub[["geo_code", "departement", "code_region"]],
        on="geo_code",
        how="left"
    )

    # Renommer colonnes pour clarté (si nécessaire)
    # On suppose que fact a au moins 'specialite' et 'charge_pathologique'
    df = df.rename(columns={
        "specialite": "specialite",
        "charge_pathologique": "charge_pathologique"
    })

    return df

df = load_data()

if df is None:
    st.error(f"Fichiers manquants. Veuillez placer {DIM_PATH} et {FACT_PATH} dans le dossier de l'application.")
    st.stop()

# Sécurité : on enlève les lignes sans département
df = df.dropna(subset=["departement"])

# -----------------------------
# Filtres dans la sidebar
# -----------------------------
st.sidebar.header("⚙️ Filtres")

# Calculer listes dynamiques
all_departements = sorted(df["departement"].dropna().unique().tolist())
all_specialites = sorted(df["specialite"].dropna().unique().tolist())
all_regions = sorted(df["code_region"].dropna().unique().tolist())

# FILTRES: par défaut tout sélectionné (plus pratique en exploration)
departement_sel = st.sidebar.multiselect(
    "Départements",
    options=all_departements,
    default=all_departements
)

specialite_sel = st.sidebar.multiselect(
    "Spécialités",
    options=all_specialites,
    default=all_specialites
)

region_sel = st.sidebar.multiselect(
    "Régions (code_region)",
    options=all_regions,
    default=all_regions
)

top_n = st.sidebar.slider("Afficher top N (par département / spécialité)", min_value=5, max_value=50, value=15, step=1)

# Application des filtres
df_filtre = df.copy()

if departement_sel:
    df_filtre = df_filtre[df_filtre["departement"].isin(departement_sel)]

if specialite_sel:
    df_filtre = df_filtre[df_filtre["specialite"].isin(specialite_sel)]

if region_sel:
    df_filtre = df_filtre[df_filtre["code_region"].isin(region_sel)]

st.sidebar.write(f"🔎 Lignes après filtre : {len(df_filtre):,}".replace(",", " "))

# -----------------------------
# Agrégations simples
# -----------------------------
agg_dep = (
    df_filtre
    .groupby("departement", as_index=False)["charge_pathologique"]
    .sum()
    .sort_values("charge_pathologique", ascending=False)
)

agg_spec = (
    df_filtre
    .groupby("specialite", as_index=False)["charge_pathologique"]
    .sum()
    .sort_values("charge_pathologique", ascending=False)
)

# Limiter pour affichage
agg_dep_top = agg_dep.head(top_n)
agg_spec_top = agg_spec.head(top_n)

# -----------------------------
# Affichage principal
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏥 Charge par département")
    st.dataframe(agg_dep_top.reset_index(drop=True), use_container_width=True)
    if not agg_dep_top.empty:
        st.bar_chart(agg_dep_top.set_index("departement")["charge_pathologique"])

with col2:
    st.subheader("🩺 Charge par spécialité")
    st.dataframe(agg_spec_top.reset_index(drop=True), use_container_width=True)
    if not agg_spec_top.empty:
        st.bar_chart(agg_spec_top.set_index("specialite")["charge_pathologique"])

# -----------------------------
# Actions utiles
# -----------------------------
st.markdown("### Actions")
csv_bytes = df_filtre.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Télécharger les données filtrées (CSV)",
    data=csv_bytes,
    file_name="filtered_data.csv",
    mime="text/csv"
)

# -----------------------------
# Détail des lignes filtrées
# -----------------------------
st.subheader("📄 Détail des données filtrées")
st.dataframe(df_filtre.reset_index(drop=True), use_container_width=True)