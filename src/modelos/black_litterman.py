"""
Black-Litterman Portfolio Optimization — Fortal Gradient
IBrX-100 | Quality-Adjusted Value Factor + LLM Sentiment

Fluxo:
  1. Prior de equilíbrio: retornos implícitos do IBrX-100 (CAPM reverso)
  2. Views: retornos esperados da regressão OLS+HAC por empresa
  3. Matriz de incerteza Ω: derivada dos p-valores e R² da regressão
  4. Black-Litterman: combina prior + views → retornos esperados ajustados
  5. Otimização MVO: pesos ótimos com restrições (long-only, peso máx 20%)

Saída:
  data/processed/pesos_bl.csv      — pesos por empresa por trimestre
  outputs/ranking_bl.csv           — ranking final com decomposição de fatores

Referência: Black & Litterman (1992), Lee et al. (2025)
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy.optimize import minimize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ          = Path(__file__).parent.parent.parent
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_OUTPUTS   = RAIZ / "outputs"
DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)

CSV_DATASET   = DIR_PROCESSED / "dataset_final.csv"
CSV_BETAS     = DIR_PROCESSED / "betas_regressao.csv"   # output do motor de fatores
CSV_PESOS     = DIR_PROCESSED / "pesos_bl.csv"
CSV_RANKING   = DIR_OUTPUTS   / "ranking_bl.csv"

# Parâmetros Black-Litterman
TAU    = 1.0    # tau alto = mais peso nas views
DELTA  = 2.5    # coeficiente de aversão ao risco (padrão BL)

# Restrições do portfólio
PESO_MIN = 0.0   # long-only
PESO_MAX = 0.20  # máximo 20% por ação


# ─────────────────────────────────────────────
# 1. PRIOR DE EQUILÍBRIO
# ─────────────────────────────────────────────

def calcular_prior_equilibrio(retornos: pd.DataFrame,
                               pesos_mercado: np.ndarray) -> np.ndarray:
    """
    Retornos implícitos de equilíbrio via CAPM reverso.
    π = δ × Σ × w_mkt
    onde Σ é a matriz de covariância dos retornos históricos.
    """
    Sigma = retornos.cov().values * 252  # anualizada
    pi    = DELTA * Sigma @ pesos_mercado
    return pi, Sigma


# ─────────────────────────────────────────────
# 2. VIEWS DA REGRESSÃO
# ─────────────────────────────────────────────

def construir_views(df_periodo: pd.DataFrame,
                    tickers: list[str]) -> tuple:
    """
    Constrói views do BL usando retornos esperados em escala real
    (betas estimados sem z-score, coluna retorno_esperado_bl).
    Se não disponível, usa rank percentil como fallback.
    """
    prior_medio = 0.074
    sigma_prior = prior_medio * 0.5

    # Verifica se temos retornos absolutos em escala real
    usa_absoluto = "retorno_esperado_bl" in df_periodo.columns and \
                   df_periodo["retorno_esperado_bl"].notna().any()

    col = "retorno_esperado_bl" if usa_absoluto else "retorno_esperado"

    ret_map = {}
    for _, row in df_periodo.iterrows():
        t = row["ticker"]
        if t in tickers and pd.notna(row.get(col)):
            ret_map[t] = float(row[col])

    rets_series = pd.Series(
        [ret_map.get(t, np.nan) for t in tickers], index=tickers)

    if usa_absoluto:
        # Views absolutas — clippa extremos para estabilidade numérica
        Q_raw = rets_series.fillna(rets_series.median())
        p5, p95 = Q_raw.quantile(0.05), Q_raw.quantile(0.95)
        Q_vals = Q_raw.clip(lower=p5, upper=p95).values
    else:
        # Fallback: rank percentil convertido para escala do prior
        rank_pct = rets_series.rank(pct=True, na_option="keep").fillna(0.5)
        Q_vals   = (prior_medio + (rank_pct.values - 0.5) * 2 * sigma_prior)

    P_rows, omega_diag = [], []
    for i in range(len(tickers)):
        p_row    = np.zeros(len(tickers))
        p_row[i] = 1.0
        P_rows.append(p_row)
        omega_diag.append(TAU * 0.01)

    return np.array(P_rows), Q_vals, np.diag(omega_diag)


# ─────────────────────────────────────────────
# 3. BLACK-LITTERMAN
# ─────────────────────────────────────────────

def black_litterman(pi: np.ndarray, Sigma: np.ndarray,
                     P: np.ndarray, Q: np.ndarray,
                     Omega: np.ndarray) -> np.ndarray:
    """
    Fórmula de Black-Litterman:
    μ_BL = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ × [(τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q]

    Combina o prior de equilíbrio (π) com as views da regressão (Q)
    ponderando pela confiança em cada fonte (τΣ e Ω).
    """
    tau_Sigma     = TAU * Sigma
    tau_Sigma_inv = np.linalg.inv(tau_Sigma)
    Omega_inv     = np.linalg.inv(Omega)

    M1 = tau_Sigma_inv + P.T @ Omega_inv @ P
    M2 = tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q

    mu_bl = np.linalg.solve(M1, M2)
    return mu_bl


# ─────────────────────────────────────────────
# 4. OTIMIZAÇÃO MVO
# ─────────────────────────────────────────────

def otimizar_pesos(mu_bl: np.ndarray,
                   Sigma: np.ndarray,
                   tickers: list[str]) -> pd.Series:
    """
    Mean-Variance Optimization com retornos BL.
    Maximiza: μ_BL^T w - (δ/2) w^T Σ w
    Sujeito a: Σw = 1, 0 ≤ w ≤ PESO_MAX
    """
    n = len(mu_bl)

    def objetivo(w):
        retorno = mu_bl @ w
        risco   = 0.5 * DELTA * w @ Sigma @ w
        return -(retorno - risco)  # negativo para minimizar

    restricoes = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    limites    = [(PESO_MIN, PESO_MAX)] * n
    w0         = np.ones(n) / n  # pesos iguais como ponto inicial

    resultado = minimize(
        objetivo, w0,
        method="SLSQP",
        bounds=limites,
        constraints=restricoes,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not resultado.success:
        log.warning(f"  Otimização não convergiu: {resultado.message}")
        return pd.Series(np.ones(n) / n, index=tickers)

    pesos = pd.Series(resultado.x, index=tickers)
    return pesos.round(4)


# ─────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def carregar_retornos_historicos(tickers: list[str],
                                  periodo_anos: int = 3) -> pd.DataFrame:
    """Baixa retornos diários históricos para calcular a matriz de covariância."""
    log.info("  Baixando retornos históricos para matriz Σ...")
    precos = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker + ".SA").history(
                period=f"{periodo_anos}y", auto_adjust=True)
            if not hist.empty:
                precos[ticker] = hist["Close"]
        except Exception as e:
            log.warning(f"  {ticker}: {e}")

    df_precos  = pd.DataFrame(precos).dropna(how="all")
    df_retornos = df_precos.pct_change().dropna(how="all")
    return df_retornos


def main():
    log.info("=" * 60)
    log.info("BLACK-LITTERMAN — CARCARÁ")
    log.info("=" * 60)

    # Carrega dataset com retornos esperados (output do factor_builder)
    CSV_RET = DIR_PROCESSED / "retornos_esperados.csv"
    if not CSV_RET.exists():
        log.error(f"Não encontrado: {CSV_RET}")
        log.error("Rode src/fatores/factor_builder.py primeiro.")
        return

    df = pd.read_csv(CSV_RET, encoding="utf-8-sig")
    log.info(f"Dataset: {len(df)} registros | {df['ticker'].nunique()} empresas")

    # Tickers únicos
    tickers = sorted(df["ticker"].unique().tolist())
    n       = len(tickers)
    log.info(f"Empresas no modelo: {n}")

    # Trimestres disponíveis
    periodos = sorted(df["periodo"].unique().tolist())
    log.info(f"Períodos: {periodos}")

    # Retornos históricos para Σ
    retornos_hist = carregar_retornos_historicos(tickers)
    tickers_ok    = [t for t in tickers if t in retornos_hist.columns]
    retornos_hist = retornos_hist[tickers_ok]

    # Pesos reais do IBrX-100 (carteira do dia 05/08/26)
    PESOS_IBX = {
        "VALE3":  0.1210, "ITUB4":  0.0837, "PETR4":  0.0696,
        "AXIA3":  0.0430, "SBSP3":  0.0365, "BBDC4":  0.0364,
        "ITSA4":  0.0303, "B3SA3":  0.0291, "WEGE3":  0.0271,
        "ABEV3":  0.0252, "BPAC11": 0.0252, "EMBJ3":  0.0239,
        "BBAS3":  0.0222, "ENEV3":  0.0193, "RENT3":  0.0142,
    }
    pesos_raw = np.array([PESOS_IBX.get(t, 1/len(tickers_ok)) for t in tickers_ok])
    pesos_mkt = pesos_raw / pesos_raw.sum()  # normaliza para somar 1
    log.info(f"Pesos IBrX-100 aplicados (soma={pesos_mkt.sum():.4f})")

    # Prior de equilíbrio
    pi, Sigma = calcular_prior_equilibrio(retornos_hist, pesos_mkt)
    log.info(f"Prior π calculado (média: {pi.mean():.4f})")

    # Processa cada período
    todos_pesos   = []
    todos_ranking = []

    for periodo in periodos:
        log.info(f"\n  Período: {periodo}")
        df_per = df[df["periodo"] == periodo].copy()

        if len(df_per) < 3:
            log.warning(f"  Poucos dados ({len(df_per)} empresas) — pulando")
            continue

        # Filtra para tickers com retornos históricos
        df_per = df_per[df_per["ticker"].isin(tickers_ok)]
        tickers_per = sorted(df_per["ticker"].unique().tolist())

        if len(tickers_per) < 2:
            continue

        # Submatriz Σ para esse período
        Sigma_per = retornos_hist[tickers_per].cov().values * 252
        pi_per    = DELTA * Sigma_per @ (np.ones(len(tickers_per)) / len(tickers_per))

        # Views da regressão (rank percentil → escala do prior)
        P, Q, Omega = construir_views(df_per, tickers_per)

        if P is None:
            log.warning(f"  Sem views para {periodo}")
            continue

        # Diagnóstico
        log.info(f"  Prior π (média): {pi_per.mean():.4f}")
        log.info(f"  Views Q (média): {Q.mean():.4f} | min={Q.min():.4f} | max={Q.max():.4f}")

        # Black-Litterman
        try:
            mu_bl = black_litterman(pi_per, Sigma_per, P, Q, Omega)
        except np.linalg.LinAlgError as e:
            log.warning(f"  Erro matricial em {periodo}: {e}")
            continue

        # Otimização MVO
        pesos = otimizar_pesos(mu_bl, Sigma_per, tickers_per)

        # Registra pesos
        for ticker, peso in pesos.items():
            todos_pesos.append({
                "periodo": periodo, "ticker": ticker, "peso_bl": peso})

        # Ranking por retorno esperado BL
        ranking = pd.DataFrame({
            "periodo":  periodo,
            "ticker":   tickers_per,
            "mu_bl":    mu_bl,
            "peso_bl":  pesos.values,
        }).sort_values("mu_bl", ascending=False).reset_index(drop=True)
        ranking["rank"] = ranking.index + 1

        # Adiciona fatores do período para decomposição
        for fator in ["pl_ratio","pvp","ev_ebitda","roe","divida_pl","score_norm"]:
            ranking = ranking.merge(
                df_per[["ticker",fator]], on="ticker", how="left")

        todos_ranking.append(ranking)

        log.info(f"  Top 5: {ranking['ticker'].head(5).tolist()}")
        log.info(f"  Pesos: min={pesos.min():.3f} max={pesos.max():.3f}")

    # Salva
    if todos_pesos:
        df_pesos = pd.DataFrame(todos_pesos)
        df_pesos.to_csv(CSV_PESOS, index=False, encoding="utf-8-sig")
        log.info(f"\nPesos salvos: {CSV_PESOS}")

    if todos_ranking:
        df_ranking = pd.concat(todos_ranking, ignore_index=True)
        df_ranking.to_csv(CSV_RANKING, index=False, encoding="utf-8-sig")
        log.info(f"Ranking salvo: {CSV_RANKING}")

        # Preview do último período
        ultimo = df_ranking[df_ranking["periodo"] == periodos[-1]]
        print(f"\nRanking {periodos[-1]}:")
        print(ultimo[["rank","ticker","mu_bl","peso_bl"]].to_string(index=False))

    return df_pesos if todos_pesos else None


if __name__ == "__main__":
    main()