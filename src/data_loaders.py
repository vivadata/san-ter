import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st

# Robust data loaders used by streamlit_unified.py

@st.cache_data
def read_csv_flexible(path_or_buffer):
    """Try several common CSV formats and return a DataFrame."""
    try:
        return pd.read_csv(path_or_buffer)
    except Exception:
        pass
    try:
        return pd.read_csv(path_or_buffer, sep=';', decimal=',', engine='python')
    except Exception:
        pass
    try:
        return pd.read_csv(path_or_buffer, sep=',', decimal='.', engine='python')
    except Exception:
        return pd.read_csv(path_or_buffer, engine='python')


@st.cache_data
def load_rpps(path: str = None):
    candidates = [
        "src/rpps_stream/rpps_long_clean.csv",
        "notebooks/FV_medecins_clusters_viz.csv",
    ]
    if path:
        candidates.insert(0, path)
    for p in candidates:
        fp = os.path.join(os.getcwd(), p)
        if os.path.exists(fp):
            try:
                df = read_csv_flexible(fp)
                for c in ["dept_code", "dept_code_clean"]:
                    if c in df.columns:
                        df[c] = df[c].astype(str).str.strip()
                if 'densite' in df.columns:
                    df['densite'] = df['densite'].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '', regex=False)
                    df['densite'] = pd.to_numeric(df['densite'], errors='coerce')
                return df
            except Exception:
                continue
    return pd.DataFrame()


@st.cache_data
def load_departements(path: str = None):
    candidates = [
        "dev2/departements.geojson",
        "notebooks/departements-100m.geojson",
        "notebooks/departements.geojson",
    ]
    if path:
        candidates.insert(0, path)
    for p in candidates:
        fp = os.path.join(os.getcwd(), p)
        if os.path.exists(fp):
            try:
                gdf = gpd.read_file(fp)
                if 'code' in gdf.columns:
                    gdf['code'] = gdf['code'].astype(str).str.strip()
                    def pad_code(c):
                        if isinstance(c, str) and c.isdigit():
                            return c.zfill(2)
                        return c
                    gdf['code'] = gdf['code'].apply(pad_code)
                return gdf.to_crs(epsg=4326)
            except Exception:
                continue
    return gpd.GeoDataFrame()


@st.cache_data
def load_dim_fact():
    candidates = [
        ("notebooks/dim_geo_departement.csv", "notebooks/fact_dep_specialite_patho.csv"),
        ("src/dim_geo_departement.csv", "src/fact_dep_specialite_patho.csv"),
    ]
    for dim_p, fact_p in candidates:
        dim_fp = os.path.join(os.getcwd(), dim_p)
        fact_fp = os.path.join(os.getcwd(), fact_p)
        if os.path.exists(dim_fp) and os.path.exists(fact_fp):
            try:
                dim = read_csv_flexible(dim_fp)
                fact = read_csv_flexible(fact_fp)
                return dim, fact
            except Exception:
                continue
    return None, None


@st.cache_data
def load_quentin_tables(base_path: str = None):
    base = Path(base_path) if base_path else Path(os.getcwd()) / 'notebooks' / 'Quentin'
    candidates = {
        'offre': ['FV_offre_pathologie_clusters.csv', 'offre_pathologie.csv'],
        'med': ['FV_medecins_clusters_viz.csv', 'offre_medical.csv'],
        'dept': ['pathologie par departement.csv']
    }
    results = {}
    for k, files in candidates.items():
        df = None
        for f in files:
            p = base / f
            if p.exists():
                try:
                    df = read_csv_flexible(p)
                    break
                except Exception:
                    continue
        results[k] = df
    gj = None
    for g in [base / 'departements-100m.geojson', base / 'departements.geojson', Path('dev2') / 'departements.geojson']:
        if g.exists():
            try:
                gj = gpd.read_file(g).to_crs(epsg=4326)
                break
            except Exception:
                continue
    results['gdf'] = gj
    return results
