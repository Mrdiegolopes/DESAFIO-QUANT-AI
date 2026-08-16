"""
Backtest Out-of-Sample — CARCARÁ
Valida o modelo comparando o ranking previsto com o retorno realizado.

Metodologia:
  - Janela expansível: estima betas com dados até T, aplica em T+1
  - Métrica principal: Spearman rank correlation (ranking previsto vs. realizado)
  - Compara full-sample vs. out-of-sample para expor inflação de métrica
  - Benchmarks: IBrX-100 equal weight, portfólio BL

Saída:
  outputs/backtest_resultados.csv
  outputs/backtest_resumo.txt
"""

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ        = Path(__file__).parent.parent.parent
DIR_PROC    = RAIZ / "data" / "processed"
DIR_OUT     = RAIZ / "outputs"
DIR_OUT.mkdir(exist_ok=True)

CSV_DATASET = DIR_PROC / "dataset_final.csv"
CSV_PESOS   = DIR_PROC / "pesos_bl.csv"

FATORES = ["pl_ratio", "pvp", "ev_ebitda", "roe", "divida_pl", "score_norm"]

# Ordem cronológica dos períodos
ORDEM_PERIODOS = [
    "1T23","2T23","3T23",
    "1T24","2T24","3T24",
    "1T25","2T25","3T25",
]


def spearman_periodo(df_periodo: pd.DataFrame,
                     col_pred: str = "retorno_esperado",
                     col_real: str = "retorno_futuro") -> float:
    """Correlação de Spearman entre ranking previsto e retorno realizado."""
    sub = df_periodo[[col_pred, col_real]].dropna()
    if len(sub) < 4:
        return np.nan
    corr, _ = spearmanr(sub[col_pred], sub[col_real])
    return round(float(corr), 4)


def rodar_regressao_simples(df_train: pd.DataFrame) -> dict:
    """OLS+HAC sobre os dados de treino. Retorna betas."""
    df_reg = df_train[FATORES + ["retorno_futuro"]].dropna()
    if len(df_reg) < len(FATORES) + 3:
        return {}
    X = sm.add_constant(df_reg[FATORES])
    y = df_reg["retorno_futuro"]
    try:
        modelo = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        return {f: float(modelo.params[f]) for f in FATORES if f in modelo.params}
    except Exception:
        return {}


def calcular_retorno_esperado_oos(row: pd.Series, betas: dict) -> float:
    """Retorno esperado out-of-sample com betas do período anterior."""
    ret = 0.0
    for f, b in betas.items():
        val = row.get(f, np.nan)
        if pd.notna(val) and pd.notna(b):
            ret += b * val
    return ret


def main():
    log.info("=" * 60)
    log.info("BACKTEST OUT-OF-SAMPLE — CARCARÁ")
    log.info("=" * 60)

    df = pd.read_csv(CSV_DATASET, encoding="utf-8-sig")
    log.info(f"Dataset: {len(df)} registros | {df['ticker'].nunique()} empresas")

    periodos = [p for p in ORDEM_PERIODOS if p in df["periodo"].unique()]
    log.info(f"Períodos: {periodos}")

    # ── Full-sample Spearman (baseline — usa todos os dados) ──
    log.info("\n[1] Spearman Full-Sample (inflacionado)...")
    df_fs = df.dropna(subset=FATORES + ["retorno_futuro"])
    betas_fs = rodar_regressao_simples(df_fs)
    if betas_fs:
        df_fs = df_fs.copy()
        df_fs["pred_fs"] = df_fs.apply(
            lambda r: calcular_retorno_esperado_oos(r, betas_fs), axis=1)
        spearman_fs = spearman_periodo(df_fs, "pred_fs", "retorno_futuro")
        log.info(f"  Spearman full-sample: {spearman_fs:.4f}")
    else:
        spearman_fs = np.nan
        log.warning("  Dados insuficientes para full-sample")

    # ── Out-of-Sample: janela expansível ──
    log.info("\n[2] Spearman Out-of-Sample (janela expansível)...")
    resultados_oos = []
    min_periodos_treino = 2  # mínimo de períodos para treinar

    for i, periodo_teste in enumerate(periodos):
        if i < min_periodos_treino:
            log.info(f"  {periodo_teste}: pulando (dados de treino insuficientes)")
            continue

        # Treina com todos os períodos ANTERIORES ao teste
        periodos_treino = periodos[:i]
        df_train = df[df["periodo"].isin(periodos_treino)].copy()
        df_test  = df[df["periodo"] == periodo_teste].copy()

        if df_train.empty or df_test.empty:
            continue

        # Estima betas só com dados de treino (nunca vê o futuro)
        betas_oos = rodar_regressao_simples(df_train)
        if not betas_oos:
            log.warning(f"  {periodo_teste}: betas insuficientes")
            continue

        # Aplica betas no período de teste
        df_test["pred_oos"] = df_test.apply(
            lambda r: calcular_retorno_esperado_oos(r, betas_oos), axis=1)

        spearman_oos = spearman_periodo(df_test, "pred_oos", "retorno_futuro")

        # Retorno do portfólio BL vs. equal weight
        # Retorno portfólio BL
        ret_bl = np.nan
        if CSV_PESOS.exists():
            pesos_bl = pd.read_csv(CSV_PESOS, encoding="utf-8-sig")
            pesos_per = pesos_bl[pesos_bl["periodo"] == periodo_teste]
            if not pesos_per.empty:
                merged = df_test.merge(pesos_per[["ticker","peso_bl"]],
                                       on="ticker", how="left")
                merged["peso_bl"] = merged["peso_bl"].fillna(0)
                ret_bl = (merged["retorno_futuro"] * merged["peso_bl"]).sum()

        # Retorno portfólio Long Top 5 (ranking direto da regressão)
        top5 = df_test.nlargest(5, "pred_oos")
        ret_top5 = top5["retorno_futuro"].mean() \
            if top5["retorno_futuro"].notna().any() else np.nan

        ret_ew = df_test["retorno_futuro"].mean()

        resultados_oos.append({
            "periodo":          periodo_teste,
            "n_treino":         len(periodos_treino),
            "spearman_oos":     spearman_oos,
            "retorno_bl":       round(ret_bl,   4) if pd.notna(ret_bl)   else np.nan,
            "retorno_top5":     round(ret_top5, 4) if pd.notna(ret_top5) else np.nan,
            "retorno_ew":       round(ret_ew,   4) if pd.notna(ret_ew)   else np.nan,
            "alpha_bl_vs_ew":   round(ret_bl   - ret_ew, 4)
                                if pd.notna(ret_bl)   and pd.notna(ret_ew) else np.nan,
            "alpha_top5_vs_ew": round(ret_top5 - ret_ew, 4)
                                if pd.notna(ret_top5) and pd.notna(ret_ew) else np.nan,
        })

        log.info(f"  {periodo_teste}: Spearman={spearman_oos:.4f} | "
                 f"Top5={ret_top5:.3f} | BL={ret_bl:.3f} | EW={ret_ew:.3f}")

    # ── Resumo ──
    df_oos = pd.DataFrame(resultados_oos)

    if df_oos.empty:
        log.warning("Sem resultados out-of-sample suficientes.")
        return

    spearman_oos_medio  = df_oos["spearman_oos"].mean()
    ret_bl_medio        = df_oos["retorno_bl"].mean()
    ret_top5_medio      = df_oos["retorno_top5"].mean()
    ret_ew_medio        = df_oos["retorno_ew"].mean()
    alpha_bl_medio      = df_oos["alpha_bl_vs_ew"].mean()
    alpha_top5_medio    = df_oos["alpha_top5_vs_ew"].mean()

    log.info(f"\n{'='*60}")
    log.info("RESUMO DO BACKTEST")
    log.info(f"{'='*60}")
    log.info(f"Spearman full-sample (inflacionado): {spearman_fs:.4f}")
    log.info(f"Spearman OOS médio:                  {spearman_oos_medio:.4f}")
    log.info(f"Degradação (inflação de métrica):    {spearman_fs - spearman_oos_medio:.4f}")
    log.info(f"Retorno Top 5 médio por trimestre:   {ret_top5_medio:.4f}")
    log.info(f"Retorno BL médio por trimestre:      {ret_bl_medio:.4f}")
    log.info(f"Retorno Equal Weight médio:          {ret_ew_medio:.4f}")
    log.info(f"Alpha Top 5 vs EW médio:             {alpha_top5_medio:.4f}")
    log.info(f"Alpha BL vs EW médio:                {alpha_bl_medio:.4f}")

    # Salva
    df_oos.to_csv(DIR_OUT / "backtest_resultados.csv",
                  index=False, encoding="utf-8-sig")

    resumo = f"""BACKTEST OUT-OF-SAMPLE — CARCARÁ
{'='*50}
Spearman full-sample (inflacionado): {spearman_fs:.4f}
Spearman OOS médio:                  {spearman_oos_medio:.4f}
Degradação (inflação de métrica):    {spearman_fs - spearman_oos_medio:.4f}

Retorno BL médio por trimestre:      {ret_bl_medio:.4f} ({ret_bl_medio*4:.2%}/ano estimado)
Retorno Equal Weight médio:          {ret_ew_medio:.4f} ({ret_ew_medio*4:.2%}/ano estimado)
Alpha BL vs EW médio:                {alpha_bl_medio:.4f}

Períodos OOS testados: {len(df_oos)}
Período de treino mínimo: {min_periodos_treino} trimestres

Nota: com N=15 empresas e ~7-9 períodos OOS, os resultados devem
ser interpretados como prova de conceito metodológico, não como
estimativa precisa de performance futura.
"""
    (DIR_OUT / "backtest_resumo.txt").write_text(resumo, encoding="utf-8")
    log.info(f"\nResultados salvos em outputs/")

    print("\n" + resumo)
    print("\nPor período:")
    print(df_oos.to_string(index=False))

    return df_oos


if __name__ == "__main__":
    main()