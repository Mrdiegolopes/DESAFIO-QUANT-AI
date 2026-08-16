"""
Corrige preços faltantes para BBAS3, EMBR3, PETR4, SBSP3.
O problema foi que yf.download() com múltiplos tickers às vezes
retorna DataFrame com estrutura diferente para alguns tickers.
Este script baixa cada um individualmente e preenche o CSV.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

RAIZ = Path(__file__).parent.parent.parent
CSV  = RAIZ / "data" / "processed" / "fatores_fundamentalistas.csv"

TRI_MES_FIM = {"1T": (3, 31), "2T": (6, 30), "3T": (9, 30)}
ANOS        = [2023, 2024, 2025]
TRIMESTRES  = ["1T", "2T", "3T"]

TICKERS_FALTANTES = ["BBAS3", "EMBR3", "PETR4", "SBSP3"]


def preco_trimestre(ticker: str, ano: int, tri: str) -> float:
    mes, dia = TRI_MES_FIM[tri]
    try:
        hist = yf.Ticker(ticker + ".SA").history(
            start=f"{ano}-{mes:02d}-01",
            end=f"{ano}-{mes:02d}-{dia}",
            auto_adjust=True
        )
        if hist.empty:
            # Tenta um intervalo maior
            hist = yf.Ticker(ticker + ".SA").history(
                start=f"{ano}-01-01",
                end=f"{ano}-{mes:02d}-{dia}",
                auto_adjust=True
            )
        return float(hist["Close"].iloc[-1]) if not hist.empty else np.nan
    except Exception as e:
        print(f"  Erro {ticker} {ano} {tri}: {e}")
        return np.nan


def main():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    print(f"CSV carregado: {len(df)} registros")
    print(f"Sem preço antes: {df['preco'].isna().sum()}")

    for ticker in TICKERS_FALTANTES:
        print(f"\nBaixando preços {ticker}...")
        for ano in ANOS:
            for tri in TRIMESTRES:
                mask = (df["ticker"] == ticker) & \
                       (df["ano"] == ano) & \
                       (df["trimestre"] == tri)
                if not mask.any():
                    continue

                preco = preco_trimestre(ticker, ano, tri)
                df.loc[mask, "preco"] = round(preco, 2) if np.isfinite(preco) else np.nan
                print(f"  {ano} {tri}: R${preco:.2f}" if np.isfinite(preco)
                      else f"  {ano} {tri}: N/D")

    print(f"\nSem preço depois: {df['preco'].isna().sum()}")
    df.to_csv(CSV, index=False, encoding="utf-8-sig")
    print(f"CSV atualizado: {CSV}")


if __name__ == "__main__":
    main()