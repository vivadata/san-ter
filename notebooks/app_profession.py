import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Test Professions",
    layout="wide"
)

st.title("👷‍♂️ Test app - Professions")

st.write("On vérifie juste que les fichiers se chargent correctement et que Streamlit affiche quelque chose.")

@st.cache_data
def load_data():
    # Charge les deux fichiers tels quels
    df_pop = pd.read_csv("DS_RP_POPULATION_COMP_2022_profession_data_prepared.csv")
    df_cat = pd.read_csv("Category_Profession.csv", sep=";")
    return df_pop, df_cat

st.write("⏳ Chargement des données...")
df_pop, df_cat = load_data()
st.write("✅ Fichiers chargés !")

st.subheader("Aperçu - Population (5 premières lignes)")
st.dataframe(df_pop.head(), use_container_width=True)

st.subheader("Aperçu - Catégories de profession (5 premières lignes)")
st.dataframe(df_cat.head(), use_container_width=True)
