"""
Motor de Fatores — Fortal Gradient
Quality-Adjusted Value Factor + LLM Sentiment

Regressão OLS com erros HAC (Newey-West) por empresa.
Variável dependente: retorno_futuro (t+1)
Fatores: P/L, P/VP, EV/EBITDA, ROE, Dívida/PL, Sentimento normalizado

Equação:
  Retorno(t+1) = β₁×P/L(t) + β₂×P/VP(t) + β₃×EV/EBITDA(t)
               + β₄×ROE(t) + β₅×Dívida/PL(t) + β₆×Sentimento_norm(t) + ε

Saída:
  data/processed/betas_regressao.csv     — betas, p-valores, R²
  data/processed/retornos_esperados.csv  — retorno esperado por empresa/trimestre
  outputs/tabela_fatores.csv             — tabela completa para o relatório
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from src.fatores.winsorization import winsorize_factors_by_quarter
from src.fatores.normalizacao import zscore_cross_sectional

PASTA_PROCESSED = BASE_DIR / "data" / "processed"
PASTA_OUTPUTS   = BASE_DIR / "outputs"
PASTA_OUTPUTS.mkdir(exist_ok=True)

# Fatores do modelo
FATORES = ["pl_ratio", "pvp", "ev_ebitda", "roe", "divida_pl", "score_norm"]

# Nomes amigáveis para o relatório
NOMES_FATORES = {
    "pl_ratio":   "P/L",
    "pvp":        "P/VP",
    "ev_ebitda":  "EV/EBITDA",
    "roe":        "ROE",
    "divida_pl":  "Dívida/PL",
    "score_norm": "Sentimento (norm.)",
}


def rodar_regressao_ols_hac(df: pd.DataFrame,
                             fatores: list[str]) -> dict:
    """
    Regressão OLS com erros HAC (Newey-West) para corrigir
    heterocedasticidade e autocorrelação em séries trimestrais.
    Retorna dict com betas, p-valores, t-stats, R², DW.
    """
    df_reg = df[fatores + ["retorno_futuro"]].dropna()

    if len(df_reg) < len(fatores) + 3:
        return {}

    X = sm.add_constant(df_reg[fatores])
    y = df_reg["retorno_futuro"]

    modelo = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 2},  # Newey-West com 2 lags (padrão para trimestral)
    )

    resultado = {
        "r2":          round(modelo.rsquared, 4),
        "r2_adj":      round(modelo.rsquared_adj, 4),
        "n_obs":       int(modelo.nobs),
        "durbin_watson": round(durbin_watson(modelo.resid), 3),
        "f_pvalue":    np.nan,  # f_pvalue não disponível com HAC — usa p-valores individuais
    }

    for fator in fatores:
        if fator in modelo.params.index:
            resultado[f"beta_{fator}"]   = round(float(modelo.params[fator]), 6)
            resultado[f"tstat_{fator}"]  = round(float(modelo.tvalues[fator]), 3)
            resultado[f"pvalue_{fator}"] = round(float(modelo.pvalues[fator]), 4)

    return resultado


def calcular_retorno_esperado(row: pd.Series, betas: dict) -> float:
    """Retorno esperado = soma(β_i × fator_i) para cada empresa/trimestre."""
    ret = 0.0
    for fator in FATORES:
        beta = betas.get(f"beta_{fator}", 0.0)
        val  = row.get(fator, np.nan)
        if not np.isnan(val) and not np.isnan(beta):
            ret += beta * val
    return round(ret, 6)


def main():
    print("=" * 60)
    print("MOTOR DE FATORES — CARCARÁ")
    print("Quality-Adjusted Value + LLM Sentiment | OLS+HAC")
    print("=" * 60)

    # Carrega dataset final
    arq = PASTA_PROCESSED / "dataset_final.csv"
    if not arq.exists():
        raise FileNotFoundError(
            f"{arq} não encontrado. "
            "Rode src/processamento/merge_datasets.py primeiro.")

    df = pd.read_csv(arq, encoding="utf-8-sig")
    print(f"[+] Dataset: {len(df)} registros | {df['ticker'].nunique()} empresas")

    # Remove linhas sem retorno futuro (não dá para treinar sem y)
    df_reg = df.dropna(subset=["retorno_futuro"]).copy()
    print(f"[+] Com retorno futuro: {len(df_reg)} registros")

    # ── Winsorização adicional por trimestre ──
    print("[+] Winsorização cross-sectional por trimestre...")
    df_reg = winsorize_factors_by_quarter(
        df_reg, FATORES, date_col="periodo")

    # ── Z-Score cross-sectional ──
    print("[+] Z-Score cross-sectional...")
    df_reg = zscore_cross_sectional(df_reg, FATORES, date_col="periodo")
    fatores_z = [f"z_{f}" for f in FATORES]

    # ── Regressão OLS+HAC (pooled — todas empresas juntas) ──
    print("\n[+] Rodando regressão OLS+HAC (pooled)...")

    # Cria cópia com colunas z renomeadas para os nomes originais dos fatores
    df_para_reg = df_reg.copy()
    for fator in FATORES:
        z_col = f"z_{fator}"
        if z_col in df_para_reg.columns:
            df_para_reg[fator] = df_para_reg[z_col]

    resultado_pooled = rodar_regressao_ols_hac(df_para_reg, FATORES)

    if not resultado_pooled:
        print("[!] Dados insuficientes para regressão.")
        return

    print(f"\n    R²          : {resultado_pooled['r2']}")
    print(f"    R² ajustado : {resultado_pooled['r2_adj']}")
    print(f"    N obs       : {resultado_pooled['n_obs']}")
    print(f"    Durbin-Watson: {resultado_pooled['durbin_watson']}")
    print(f"    F p-valor   : {resultado_pooled['f_pvalue']}")
    print()
    print(f"    {'Fator':<20} {'Beta':>10} {'t-stat':>10} {'p-valor':>10} {'Sig':>5}")
    print("    " + "-"*55)
    for fator in FATORES:
        beta   = resultado_pooled.get(f"beta_{fator}", np.nan)
        tstat  = resultado_pooled.get(f"tstat_{fator}", np.nan)
        pval   = resultado_pooled.get(f"pvalue_{fator}", np.nan)
        sig    = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        nome   = NOMES_FATORES.get(fator, fator)
        print(f"    {nome:<20} {beta:>10.4f} {tstat:>10.3f} {pval:>10.4f} {sig:>5}")

    # ── Salva betas ──
    registros_betas = []
    for fator in FATORES:
        registros_betas.append({
            "fator":   fator,
            "nome":    NOMES_FATORES.get(fator, fator),
            "beta":    resultado_pooled.get(f"beta_{fator}", np.nan),
            "tstat":   resultado_pooled.get(f"tstat_{fator}", np.nan),
            "pvalue":  resultado_pooled.get(f"pvalue_{fator}", np.nan),
            "r2":      resultado_pooled["r2"],
            "r2_adj":  resultado_pooled["r2_adj"],
            "n_obs":   resultado_pooled["n_obs"],
            "dw":      resultado_pooled["durbin_watson"],
        })

    df_betas = pd.DataFrame(registros_betas)
    df_betas.to_csv(PASTA_PROCESSED / "betas_regressao.csv",
                    index=False, encoding="utf-8-sig")
    print(f"\n[✔] Betas salvos: {PASTA_PROCESSED / 'betas_regressao.csv'}")

    # ── Regressão auxiliar SEM z-score para views do BL ──
    # Os betas z-score são úteis para comparar importância relativa dos fatores,
    # mas geram views na escala errada para o BL. Rodamos uma segunda regressão
    # com os fatores originais (winsorizados, não z-score) para ter betas
    # que geram retornos esperados na escala real (% de retorno).
    print("\n[+] Regressão auxiliar sem z-score (para views BL em escala real)...")
    resultado_bl = rodar_regressao_ols_hac(df_reg, FATORES)

    if resultado_bl:
        print(f"    R² (escala real): {resultado_bl['r2']:.4f}")
        for fator in FATORES:
            beta_bl = resultado_bl.get(f"beta_{fator}", np.nan)
            nome    = NOMES_FATORES.get(fator, fator)
            print(f"    {nome:<20}: beta_bl={beta_bl:.6f}")

        # Salva betas BL separadamente
        registros_bl = []
        for fator in FATORES:
            registros_bl.append({
                "fator": fator,
                "nome":  NOMES_FATORES.get(fator, fator),
                "beta":  resultado_bl.get(f"beta_{fator}", np.nan),
                "r2":    resultado_bl["r2"],
            })
        pd.DataFrame(registros_bl).to_csv(
            PASTA_PROCESSED / "betas_bl.csv",
            index=False, encoding="utf-8-sig")

        # Retorno esperado BL (escala real) — usado como view absoluta no BL
        df["retorno_esperado_bl"] = df.apply(
            lambda row: calcular_retorno_esperado(row, resultado_bl), axis=1)
        print(f"[✔] Betas BL salvos: {PASTA_PROCESSED / 'betas_bl.csv'}")
    else:
        df["retorno_esperado_bl"] = df["retorno_esperado"]
        print("[!] Regressão BL falhou — usando retorno_esperado como fallback")

    # ── Retornos esperados por empresa/trimestre ──
    betas_dict = resultado_pooled
    df["retorno_esperado"] = df.apply(
        lambda row: calcular_retorno_esperado(row, betas_dict), axis=1)

    # Rank percentil por período (backup — BL agora usa retorno_esperado_bl)
    df["rank_pct"] = df.groupby("periodo")["retorno_esperado"].rank(pct=True)

    # Ranking por período
    df["rank_periodo"] = df.groupby("periodo")["retorno_esperado"].rank(
        ascending=False, method="dense").astype(int)

    # Salva
    df.to_csv(PASTA_PROCESSED / "retornos_esperados.csv",
              index=False, encoding="utf-8-sig")

    # Tabela para o relatório
    tabela = df[["ticker","periodo","pl_ratio","pvp","ev_ebitda",
                 "roe","divida_pl","score_norm","retorno_esperado",
                 "retorno_futuro","rank_periodo"]].copy()
    tabela.to_csv(PASTA_OUTPUTS / "tabela_fatores.csv",
                  index=False, encoding="utf-8-sig")

    print(f"[✔] Retornos esperados salvos")
    print(f"[✔] Tabela para relatório salva")

    # Preview ranking último período
    ultimo_periodo = df["periodo"].max()
    rank_ult = df[df["periodo"] == ultimo_periodo].sort_values(
        "retorno_esperado", ascending=False)
    print(f"\nRanking {ultimo_periodo} (top 10):")
    print(rank_ult[["rank_periodo","ticker","retorno_esperado",
                    "roe","score_norm"]].head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    main()