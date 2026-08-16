"""
Busca os nomes corretos de SBSP3, ITSA4 e BBAS3 na CVM.
"""
import pandas as pd
from pathlib import Path

DIR_CVM = Path(r"C:\Users\55859\value-factor-model-ibrx100\data\raw\cvm")

df = pd.read_csv(DIR_CVM / "itr_cia_aberta_DRE_con_2024.csv",
                 sep=";", encoding="latin1", decimal=",", low_memory=False)

col_nome = "DENOM_CIA" if "DENOM_CIA" in df.columns else "NOME_CIA"
nomes = df[col_nome].dropna().unique()

for busca in ["SABESP", "COMPANHIA DE SANEAMENTO", "SBSP",
              "ITAUSA", "ITAÚSA", "INVESTIMENTOS ITAU",
              "BANCO DO BRASIL", "BB SEGURIDADE", "BRASIL S.A"]:
    matches = [n for n in nomes if busca.upper() in str(n).upper()]
    if matches:
        print(f"'{busca}' → {matches[:3]}")