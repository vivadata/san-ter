import streamlit as st
import pandas as pd
import altair as alt

# =========================
# 1. Chargement des données
# =========================
st.title("📊 Dev 1 – Charge pathologique par département & spécialité")

@st.cache_data
def load_data(
    fact_path: str = "fact_dep_specialite_patho.csv",
    dim_path: str = "dim_geo_departement.csv",
):
    """
    Charge les tables Dev 1 :
    - fact_dep_specialite_patho.csv : charge pathologique par dept x spécialité
    - dim_geo_departement.csv       : infos géographiques sur les départements
    """
    fact = pd.read_csv(fact_path)
    dim = pd.read_csv(dim_path)

    # Harmonisation des codes
    fact["geo_code"] = fact["geo_code"].astype(str).str.zfill(2)
    dim["geo_code"] = dim["geo_code"].astype(str).str.zfill(2)

    # Merge pour avoir libellé et région dans le fact
    df = fact.merge(
        dim[["geo_code", "geo_libelle", "code_region"]],
        on="geo_code",
        how="left",
    )

    return df, fact, dim


# =========================
# 2. App Streamlit
# =========================

def main():
    st.set_page_config(
        page_title="Dev 1 – Pathologies par département",
        layout="wide"
    )

    st.title("📊 Dev 1 – Charge pathologique par département & spécialité")

    # --- Chargement des données ---
    df, fact, dim = load_data()

    # =========================
    #   FILTRES (sidebar)
    # =========================

    st.sidebar.header("🎛️ Filtres")

    # Filtre région
    regions = dim["code_region"].dropna().unique()
    regions = sorted(regions)
    region_labels = {"ALL": "Toutes les régions"}
    region_labels.update({str(r): str(r) for r in regions})

    selected_region = st.sidebar.selectbox(
        "Région (code INSEE région)",
        options=["ALL"] + [str(r) for r in regions],
        format_func=lambda x: region_labels[x],
        index=0,
    )

    # Filtre spécialité
    specialites = sorted(df["specialite"].dropna().unique())
    selected_specialites = st.sidebar.multiselect(
        "Spécialité(s) médicale(s)",
        options=specialites,
        default=["Oncologie"] if "Oncologie" in specialites else specialites[:3],
    )

    if not selected_specialites:
        st.warning("Sélectionne au moins une spécialité pour afficher les résultats.")
        return

    # =========================
    #   FILTRAGE DES DONNÉES
    # =========================

    data = df.copy()
    data = data[data["specialite"].isin(selected_specialites)]

    if selected_region != "ALL":
        data = data[data["code_region"].astype(str) == selected_region]

    # Agrégation pour la vue principale
    agg_dep = (
        data.groupby(["geo_code", "geo_libelle"], as_index=False)["charge_pathologique"]
        .sum()
        .sort_values("charge_pathologique", ascending=False)
    )

    # =========================
    #   KPIs
    # =========================

    total_charge = agg_dep["charge_pathologique"].sum()
    nb_depts = agg_dep["geo_code"].nunique()
    nb_spe = len(selected_specialites)

    col1, col2, col3 = st.columns(3)
    col1.metric("Charge totale sélectionnée", f"{int(total_charge):,}".replace(",", " "))
    col2.metric("Nombre de départements", nb_depts)
    col3.metric("Nombre de spécialités filtrées", nb_spe)

    # =========================
    #   GRAPHIQUE PRINCIPAL
    # =========================

    st.subheader("Répartition par département")

    if agg_dep.empty:
        st.info("Aucune donnée pour ce filtre (région + spécialité).")
        return

    chart = (
        alt.Chart(agg_dep)
        .mark_bar()
        .encode(
            x=alt.X("charge_pathologique:Q", title="Charge pathologique"),
            y=alt.Y(
                "geo_libelle:N",
                sort="-x",
                title="Département"
            ),
            tooltip=[
                alt.Tooltip("geo_libelle:N", title="Département"),
                alt.Tooltip("geo_code:N", title="Code"),
                alt.Tooltip("charge_pathologique:Q", title="Charge pathologique", format=",")
            ],
        )
        .properties(
            height=500
        )
    )

    st.altair_chart(chart, use_container_width=True)

    # =========================
    #   DÉTAIL TABLEAU
    # =========================

    st.subheader("Détails par département")

    st.dataframe(
        agg_dep.reset_index(drop=True),
        use_container_width=True,
    )

    # Bouton de téléchargement
    csv_export = agg_dep.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger les données filtrées (CSV)",
        data=csv_export,
        file_name="dep_specialite_filtre.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()