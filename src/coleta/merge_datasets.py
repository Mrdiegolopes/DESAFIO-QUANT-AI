"""
Merge de Datasets — Fundamentais + Sentimento
Junta fatores_fundamentalistas.csv + scores_sentimento.csv
e gera o dataset final para a regressão OLS+HAC.

Saída: data/processed/dataset_final.csv

Tratamentos aplicados:
  1. Normalização cross-sectional do score de sentimento por trimestre
     (resolve o viés de positividade — o que importa é a variação relativa)
  2. Winsorização 1%-99% em todos os fatores
  3. Retorno futuro calculado via preços históricos (variável dependente)
  4. Dummy COVID para 1T20 e 2T20
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ          = Path(__file__).parent.parent.parent
DIR_PROCESSED = RAIZ / "data" / "processed"
CSV_FUND      = DIR_PROCESSED / "fatores_fundamentalistas.csv"
CSV_SENT      = DIR_PROCESSED / "scores_sentimento.csv"
CSV_SAIDA     = DIR_PROCESSED / "dataset_final.csv"

# Mapa trimestre → mês de início do PRÓXIMO trimestre (para calcular retorno)
PROX_TRI = {
    "1T": ("2T", 0),   # mesmo ano
    "2T": ("3T", 0),
    "3T": ("4T", 0),
    "4T": ("1T", 1),   # ano seguinte
}

TRI_MES = {"1T": 3, "2T": 6, "3T": 9, "4T": 12}


def calcular_retorno_futuro(ticker: str, ano: int,
                             trimestre: str) -> float:
    """
    Retorno do preço entre o fim do trimestre T e o fim do trimestre T+1.
    É a variável dependente da regressão.
    """
    mes_atual = TRI_MES.get(trimestre)
    if mes_atual is None:
        return np.nan

    prox_tri, delta_ano = PROX_TRI[trimestre]
    mes_prox   = TRI_MES[prox_tri]
    ano_prox   = ano + delta_ano

    try:
        ticker_sa = ticker + ".SA"
        # Preço no fim do trimestre atual
        hist_atual = yf.Ticker(ticker_sa).history(
            start=f"{ano}-{mes_atual:02d}-01",
            end=f"{ano}-{mes_atual:02d}-28",
            auto_adjust=True)

        # Preço no fim do próximo trimestre
        hist_prox = yf.Ticker(ticker_sa).history(
            start=f"{ano_prox}-{mes_prox:02d}-01",
            end=f"{ano_prox}-{mes_prox:02d}-28",
            auto_adjust=True)

        if hist_atual.empty or hist_prox.empty:
            return np.nan

        p0 = float(hist_atual["Close"].iloc[-1])
        p1 = float(hist_prox["Close"].iloc[-1])

        if p0 == 0:
            return np.nan

        return round((p1 - p0) / p0, 4)

    except Exception as e:
        log.debug(f"  Retorno {ticker} {ano} {trimestre}: {e}")
        return np.nan


def normalizar_cross_sectional(df: pd.DataFrame,
                                col: str) -> pd.Series:
    """
    Normalização cross-sectional por trimestre:
    converte o valor absoluto em percentil dentro do mesmo período.
    Resolve o viés de positividade do sentimento — o que importa
    não é o score absoluto (todos tendem a ser positivos), mas
    quem está mais positivo do que os pares no mesmo trimestre.
    """
    def _rank_periodo(grupo):
        return grupo[col].rank(pct=True)

    return df.groupby("periodo", group_keys=False).apply(_rank_periodo)


def main():
    log.info("=" * 60)
    log.info("MERGE DATASETS — Fundamentais + Sentimento")
    log.info("=" * 60)

    # ── Carrega fundamentais ──
    if not CSV_FUND.exists():
        log.error(f"Não encontrado: {CSV_FUND}")
        return
    df_fund = pd.read_csv(CSV_FUND, encoding="utf-8-sig")
    log.info(f"Fundamentais: {len(df_fund)} registros")

    # ── Carrega sentimento ──
    if not CSV_SENT.exists():
        log.error(f"Não encontrado: {CSV_SENT}")
        log.error("Rode src/sentimento/score_sentimento.py primeiro.")
        return
    df_sent = pd.read_csv(CSV_SENT, encoding="utf-8-sig")
    log.info(f"Sentimento: {len(df_sent)} registros")

    # ── Merge por ticker + ano + trimestre ──
    df = pd.merge(
        df_fund,
        df_sent[["ticker","ano","trimestre","tom","score","resumo"]],
        on=["ticker","ano","trimestre"],
        how="left",
    )
    log.info(f"Após merge: {len(df)} registros | "
             f"com sentimento: {df['score'].notna().sum()}")

    # ── Normalização cross-sectional do sentimento ──
    df["score_norm"] = normalizar_cross_sectional(df, "score")
    log.info("Score normalizado cross-sectionally por trimestre")

    # ── Dummy COVID ──
    df["dummy_covid"] = (
        ((df["ano"] == 2020) & (df["trimestre"].isin(["1T","2T"])))
    ).astype(int)

    # ── Retorno futuro (variável dependente) ──
    log.info("\nCalculando retornos futuros (yfinance)...")
    retornos = []
    for _, row in df.iterrows():
        ret = calcular_retorno_futuro(
            str(row["ticker"]), int(row["ano"]), str(row["trimestre"]))
        retornos.append(ret)
    df["retorno_futuro"] = retornos

    cobertura_ret = df["retorno_futuro"].notna().mean() * 100
    log.info(f"Cobertura retorno futuro: {cobertura_ret:.1f}%")

    # ── Winsorização final 1%-99% ──
    cols_win = ["pl_ratio","pvp","ev_ebitda","roe","divida_pl",
                "score","score_norm","retorno_futuro"]
    for col in cols_win:
        if col in df.columns and df[col].notna().sum() > 10:
            p1  = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=p1, upper=p99)

    # ── Salva ──
    df = df.sort_values(["ticker","ano","trimestre"]).reset_index(drop=True)
    df.to_csv(CSV_SAIDA, index=False, encoding="utf-8-sig")

    log.info(f"\nDataset final salvo: {CSV_SAIDA}")
    log.info(f"Total: {len(df)} registros | "
             f"{df['ticker'].nunique()} empresas")

    # ── Resumo ──
    log.info("\nCobertura das variáveis:")
    cols_check = ["pl_ratio","pvp","ev_ebitda","roe","divida_pl",
                  "score","score_norm","retorno_futuro"]
    for col in cols_check:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            st  = "✓" if pct > 70 else "⚠"
            log.info(f"  {st} {col:15}: {pct:.1f}%")

    print("\nPreview:")
    print(df[["ticker","periodo","pl_ratio","roe","score",
              "score_norm","retorno_futuro"]].head(15).to_string(index=False))

    return df


if __name__ == "__main__":
    df = main()