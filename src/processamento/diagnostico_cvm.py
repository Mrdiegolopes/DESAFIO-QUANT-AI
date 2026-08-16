"""
Diagnóstico rápido — nomes reais das empresas nos CSVs locais da CVM.
Lê os arquivos que você já baixou em data/raw/cvm/
"""
import pandas as pd
from pathlib import Path

# Ajuste esse caminho se necessário
DIR_CVM = Path(r"C:\Users\55859\value-factor-model-ibrx100\data\raw\cvm")

TERMOS = {
    "VALE3":  "VALE",
    "ITUB4":  "ITAU",
    "PETR4":  "PETROBRAS",
    "AXIA3":  "AXIA",
    "SBSP3":  "SABESP",
    "BBDC4":  "BRADESCO",
    "ITSA4":  "ITAUSA",
    "B3SA3":  "B3",
    "WEGE3":  "WEG",
    "ABEV3":  "AMBEV",
    "BPAC11": "BTG",
    "EMBR3":  "EMBRAER",
    "BBAS3":  "BANCO DO BRASIL",
    "ENEV3":  "ENEVA",
    "RENT3":  "LOCALIZA",
}

# Lê o DRE de 2024 (um ano suficiente para diagnóstico)
arquivo = DIR_CVM / "itr_cia_aberta_DRE_con_2024.csv"
print(f"Lendo: {arquivo}")
df = pd.read_csv(arquivo, sep=";", encoding="latin1",
                 decimal=",", low_memory=False)

print(f"Colunas: {list(df.columns)}")
print(f"Total de linhas: {len(df)}")

# Coluna de nome
col_nome = next((c for c in ["DENOM_CIA","NOME_CIA"] if c in df.columns), None)
print(f"Coluna de nome: {col_nome}")

if col_nome:
    nomes = df[col_nome].dropna().unique()
    print(f"Total empresas únicas: {len(nomes)}")
    print("\n--- NOMES REAIS NA CVM ---")
    for ticker, termo in TERMOS.items():
        matches = [n for n in nomes if termo.upper() in str(n).upper()]
        status = "✓" if matches else "✗ NAO ENCONTRADO"
        print(f"{ticker:8} | {termo:20} | {status} | {matches[:3]}")