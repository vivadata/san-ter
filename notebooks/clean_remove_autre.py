#!/usr/bin/env python3
"""Script de nettoyage: sauvegarde et suppression des lignes où specialite == 'Autre'."""
import shutil
from pathlib import Path
import pandas as pd

p = Path(__file__).resolve().parent / 'fact_dep_specialite_patho.csv'
if not p.exists():
    print(f"Fichier introuvable: {p}")
    raise SystemExit(1)

bak = p.with_suffix('.csv.bak')
shutil.copy2(p, bak)
print(f"Sauvegarde créée: {bak}")

df = pd.read_csv(p, dtype=str)
if 'specialite' not in df.columns:
    print('Colonne specialite introuvable. Aucun changement effectué.')
    raise SystemExit(1)

before = len(df)
# Filtrer toutes les lignes où specialite == 'Autre' (exact match)
df_clean = df[df['specialite'].astype(str).str.strip().str.lower() != 'autre']
after = len(df_clean)

if after == before:
    print('Aucune ligne supprimée (aucune valeur "Autre" trouvée).')
else:
    df_clean.to_csv(p, index=False)
    print(f'Lignes totales avant: {before}; après: {after}. Fichier mis à jour.')

print('Terminé.')
