import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px


@st.cache_data
def load_data(csv_path: str):
    # Lecture CSV avec séparateur ',' mais décimales en ',' dans la colonne 'valeur'
    df = pd.read_csv(csv_path, dtype={"departement_de_domicile": str})
    # Normaliser noms de colonnes (éventuels espaces)
    df.columns = [c.strip() for c in df.columns]
    # Standardiser les colonnes essentielles
    if "annee_de_deces" in df.columns:
        df["annee_de_deces"] = pd.to_numeric(df["annee_de_deces"], errors="coerce")
        df.rename(columns={"annee_de_deces": "annee"}, inplace=True)
    # Nettoyer la colonne 'valeur': remplacer "," par "." puis to_numeric
    if "valeur" in df.columns:
        df["valeur"] = (
            df["valeur"].astype(str)
            .str.replace(",", ".", regex=False)
            .replace("\u00a0", "", regex=True)
        )
        df["valeur"] = pd.to_numeric(df["valeur"], errors="coerce")
    return df


@st.cache_data
def load_departements(geojson_path: str):
    gdf = gpd.read_file(geojson_path)
    # Detecter automatiquement la colonne code département
    candidates = [
        "code",
        "code_insee",
        "CODDEP",
        "DEP",
        "INSEE",
        "dep",
        "num_dep",
        "NUM_DEPT",
    ]
    for c in candidates:
        if c in gdf.columns:
            gdf = gdf.rename(columns={c: "dep"})
            break
    if "dep" not in gdf.columns:
        # Chercher une colonne qui ressemble à des codes
        for c in gdf.columns:
            sample = gdf[c].astype(str).head(20).tolist()
            matches = sum(1 for s in sample if s.strip().replace(" ", "") != "" and any(ch.isdigit() for ch in s))
            if matches >= 3:
                gdf = gdf.rename(columns={c: "dep"})
                break
    gdf["dep"] = gdf["dep"].astype(str).str.zfill(2)
    return gdf


def main():
    st.set_page_config(page_title="Décès par département", layout="wide")
    st.title("Exploration des taux de mortalité (2018-2023) — par département")

    csv_path = "notebooks/data deces 2018-2023 - Données.csv"
    geojson_path = "dev2/departements.geojson"

    df = load_data(csv_path)
    gdf = load_departements(geojson_path)

    # Filtrage simple: proposer cause, âge, sexe
    cause_opts = ["Tous"] + sorted(df["cause_initiale_de_deces"].dropna().unique().tolist())
    age_opts = ["Tous"] + sorted(df["grandes_classes_age"].dropna().unique().tolist())
    sexe_opts = ["Tous"] + sorted(df["sexe"].dropna().unique().tolist())

    with st.sidebar:
        st.header("Filtres")
        # Sélecteur d'année focalisé (par défaut 2022 si présent)
        years = sorted(df["annee"].dropna().astype(int).unique().tolist())
        default_idx = years.index(2022) if 2022 in years else max(0, len(years) - 1)
        selected_year = st.selectbox("Année", years, index=default_idx)

        selected_cause = st.selectbox("Cause initiale", cause_opts)
        selected_age = st.selectbox("Tranche d'âge", age_opts)
        selected_sexe = st.selectbox("Sexe", sexe_opts)
        dep_choice = st.selectbox("Département (pour série temporelle)", ["France"] + sorted(df["departement_de_domicile"].dropna().unique().tolist()))

    # Appliquer filtres
    df_f = df.copy()
    if selected_cause != "Tous":
        df_f = df_f[df_f["cause_initiale_de_deces"] == selected_cause]
    if selected_age != "Tous":
        df_f = df_f[df_f["grandes_classes_age"] == selected_age]
    if selected_sexe != "Tous":
        df_f = df_f[df_f["sexe"] == selected_sexe]

    # KPI calculations (focalisé sur l'année sélectionnée)
    latest_year = int(selected_year)
    prev_year = latest_year - 1

    mean_latest = df_f[df_f["annee"] == latest_year]["valeur"].mean()
    mean_prev = df_f[df_f["annee"] == prev_year]["valeur"].mean()
    pct_change = None
    if pd.notna(mean_prev) and mean_prev != 0:
        pct_change = (mean_latest - mean_prev) / abs(mean_prev) * 100

    # Département le plus élevé cette année (moyenne par département)
    dept_latest = (
        df_f[df_f["annee"] == latest_year]
        .groupby("departement_de_domicile")["valeur"]
        .mean()
        .dropna()
    )
    if not dept_latest.empty:
        top_dept = dept_latest.idxmax()
        top_dept_value = dept_latest.max()
    else:
        top_dept = None
        top_dept_value = None

    # Affichage des KPI
    k1, k2, k3 = st.columns(3)
    k1.metric(label=f"Moyenne (année {latest_year})", value=f"{mean_latest:.2f}" if pd.notna(mean_latest) else "N/A")
    k2.metric(label=f"Évolution vs {prev_year}", value=(f"{pct_change:.1f}%" if pct_change is not None else "N/A"))
    k3.metric(label=f"Département max ({latest_year})", value=(f"{top_dept} — {top_dept_value:.2f}" if top_dept is not None else "N/A"))

    st.markdown("---")

    # Série temporelle
    st.subheader("Série temporelle")
    if dep_choice == "France":
        ts = df_f.groupby("annee")["valeur"].mean().reset_index()
        fig_ts = px.line(ts, x="annee", y="valeur", markers=True, title="Moyenne nationale (par année)")
    else:
        ts = (
            df_f[df_f["departement_de_domicile"] == dep_choice]
            .groupby("annee")["valeur"]
            .mean()
            .reset_index()
        )
        fig_ts = px.line(ts, x="annee", y="valeur", markers=True, title=f"Série pour le département {dep_choice}")
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # Graphique par cause
    st.subheader(f"Répartition par cause — année {latest_year}")
    # Slider pour nombre de causes à afficher
    top_n = st.sidebar.slider("Nombre de causes à afficher", min_value=5, max_value=30, value=10)

    cause_vals = (
        df_f[df_f["annee"] == latest_year]
        .groupby("cause_initiale_de_deces")["valeur"]
        .mean()
        .reset_index()
        .sort_values("valeur", ascending=False)
    )
    if cause_vals.empty:
        st.info("Aucune donnée disponible pour la sélection actuelle et l'année choisie.")
    else:
        fig_cause = px.bar(
            cause_vals.head(top_n),
            x="valeur",
            y="cause_initiale_de_deces",
            orientation="h",
            title=f"Top {top_n} causes par valeur moyenne ({latest_year})",
            labels={"valeur": "Valeur (taux)", "cause_initiale_de_deces": "Cause"},
        )
        fig_cause.update_layout(yaxis={'categoryorder':'total ascending'}, margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_cause, use_container_width=True)

    st.markdown("---")

    # Carte choroplèthe pour l'année la plus récente
    st.subheader(f"Carte choroplèthe — année {latest_year}")
    dept_vals = (
        df_f[df_f["annee"] == latest_year]
        .groupby("departement_de_domicile")["valeur"]
        .mean()
        .reset_index()
        .rename(columns={"departement_de_domicile": "dep", "valeur": "valeur_latest"})
    )

    # Normaliser codes département pour merge
    dept_vals["dep"] = dept_vals["dep"].astype(str).str.zfill(2)

    gdf2 = gdf.merge(dept_vals, on="dep", how="left")

    # Si la géométrie est valide, utiliser plotly express choropleth_mapbox
    try:
        geojson = gdf2.__geo_interface__
        fig_map = px.choropleth_mapbox(
            gdf2,
            geojson=geojson,
            locations="dep",
            color="valeur_latest",
            featureidkey="properties.dep",
            center={"lat": 46.5, "lon": 2.5},
            mapbox_style="carto-positron",
            zoom=4.5,
            color_continuous_scale="Viridis",
            title=f"Taux moyen (année {latest_year})",
        )
        fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.error(f"Impossible de générer la carte: {e}")


if __name__ == "__main__":
    main()
