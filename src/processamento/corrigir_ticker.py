import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

CSV = Path("data/processed/fatores_fundamentalistas.csv")
df = pd.read_csv(CSV, encoding="utf-8-sig")

# Renomeia EMBR3 para EMBJ3 (mesma empresa, ticker mudou com migração ao Novo Mercado)
df["ticker"] = df["ticker"].replace("EMBR3", "EMBJ3")

TRI_MES = {"1T": (3, 31), "2T": (6, 30), "3T": (9, 30)}
ANOS = [2023, 2024, 2025]

print("Baixando precos EMBJ3.SA...")
for ano in ANOS:
    for tri, (mes, dia) in TRI_MES.items():
        mask = (df["ticker"] == "EMBJ3") & (df["ano"] == ano) & (df["trimestre"] == tri)
        if not mask.any():
            continue
        try:
            hist = yf.Ticker("EMBJ3.SA").history(
                start=f"{ano}-01-01",
                end=f"{ano}-{mes:02d}-{dia}",
                auto_adjust=True
            )
            preco = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan
        except Exception as e:
            print(f"  Erro {ano} {tri}: {e}")
            preco = np.nan

        df.loc[mask, "preco"] = round(preco, 2) if np.isfinite(preco) else np.nan
        status = f"R${preco:.2f}" if np.isfinite(preco) else "N/D"
        print(f"  {ano} {tri}: {status}")

df.to_csv(CSV, index=False, encoding="utf-8-sig")
print(f"\nSem preco depois: {df['preco'].isna().sum()}")
print("CSV atualizado.")