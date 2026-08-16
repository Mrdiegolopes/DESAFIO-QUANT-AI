"""
Diagnóstico PETR4 — salva resultado em Excel para análise.
"""
import pandas as pd
from pathlib import Path

DIR = Path("data/raw/cvm")
OUT = Path("debug_petr4.xlsx")

# Carrega DRE 2024
df = pd.read_csv(
    DIR / "itr_cia_aberta_DRE_con_2024.csv",
    sep=";", encoding="latin1", decimal=",", low_memory=False
)

# Aba 1: todos os nomes que contêm PETRO
nomes = df["DENOM_CIA"].dropna().unique()
nomes_petro = [n for n in nomes if "PETRO" in str(n).upper()]
df_nomes = pd.DataFrame({"nomes_com_PETRO": nomes_petro})

# Aba 2: dados da PETR4 filtrando por nome exato
petr = df[df["DENOM_CIA"].astype(str).str.strip() == "PETROLEO BRASILEIRO S.A. PETROBRAS"]
petr_1t24 = petr[petr["DT_FIM_EXERC"].astype(str).str.strip() == "2024-03-31"]
contas_3 = petr_1t24[petr_1t24["CD_CONTA"].astype(str).str.startswith("3.")]

# Aba 3: ORDEM_EXERC disponíveis
df_ordem = pd.DataFrame({"ORDEM_EXERC": petr["ORDEM_EXERC"].unique()})

with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    df_nomes.to_excel(w, sheet_name="Nomes_PETRO", index=False)
    contas_3.to_excel(w, sheet_name="Contas_DRE_1T24", index=False)
    petr_1t24.to_excel(w, sheet_name="Todos_dados_1T24", index=False)
    df_ordem.to_excel(w, sheet_name="ORDEM_EXERC", index=False)

print(f"Salvo: {OUT.absolute()}")
print(f"Linhas PETR4 total: {len(petr)}")
print(f"Linhas PETR4 1T24:  {len(petr_1t24)}")
print(f"Contas DRE (3.x):   {len(contas_3)}")
print(f"Nomes com PETRO:    {nomes_petro}")