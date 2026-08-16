"""
Coleta de Fundamentais — CVM + yfinance
Versão definitiva

Fontes:
  - cvm_historico_dre_con.csv  → DRE trimestral (1T/2T/3T de todos os anos)
  - cvm_historico_bpa_con.csv  → Balanço Ativo trimestral
  - cvm_historico_bpp_con.csv  → Balanço Passivo trimestral
  - itr_cia_aberta_*_con_ANO   → ITRs por ano (mesmas informações, backup)
  - yfinance                   → Preço histórico + shares outstanding

Período: 2023–2025 | 15 empresas | 1T/2T/3T por ano
Nota: 4T não está disponível no ITR (só no DFP anual, não coletado).
      Usamos 3T como proxy para 4T quando necessário no modelo.

Métricas:
  Valuation: P/L, P/VP, EV/EBITDA, Dividend Yield
  Quality:   ROE, Dívida Líquida / PL

Saída: data/processed/fatores_fundamentalistas.csv
"""

import warnings
warnings.filterwarnings("ignore")

import time
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

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
RAIZ          = Path(__file__).parent.parent.parent
DIR_CVM       = RAIZ / "data" / "raw" / "cvm"
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────
TICKERS = [
    "VALE3",  "ITUB4",  "PETR4",  "AXIA3",  "SBSP3",
    "BBDC4",  "ITSA4",  "B3SA3",  "WEGE3",  "ABEV3",
    "BPAC11", "EMBR3",  "BBAS3",  "ENEV3",  "RENT3",
]

# Apenas trimestres disponíveis no ITR (4T está no DFP, não coletado)
TRIMESTRES_DISPONIVEIS = ["1T", "2T", "3T"]
ANOS = [2023, 2024, 2025]

# Data de fim de cada trimestre (para filtrar DT_FIM_EXERC)
TRI_DATA = {
    "1T": "-03-31",
    "2T": "-06-30",
    "3T": "-09-30",
}

# Nomes exatos como aparecem na CVM (validados pelo diagnóstico)
TICKER_NOME_CVM = {
    "VALE3":  "VALE S.A.",
    "ITUB4":  "ITAU UNIBANCO HOLDING S.A.",
    "PETR4":  "PETROLEO BRASILEIRO S.A. PETROBRAS",
    "AXIA3":  "AXIA ENERGIA S.A.",
    "SBSP3":  "CIA SANEAMENTO BASICO EST SAO PAULO",
    "BBDC4":  "BCO BRADESCO S.A.",
    "ITSA4":  "ITAÚSA S.A.",
    "B3SA3":  "B3 S.A. - BRASIL, BOLSA, BALCÃO",
    "WEGE3":  "WEG S.A.",
    "ABEV3":  "AMBEV S.A.",
    "BPAC11": "BCO BTG PACTUAL S.A.",
    "EMBR3":  "EMBRAER S.A.",
    "BBAS3":  "BCO BRASIL S.A.",
    "ENEV3":  "ENEVA S.A",
    "RENT3":  "LOCALIZA RENT A CAR S.A.",
}

# Contas CVM que precisamos
CONTAS = {
    "DRE": {
        "lucro_liquido":   ["3.11", "3.09", "3.08"],
        "ebit":            ["3.05", "3.07", "3.06"],
        "receita_liquida": ["3.01"],
    },
    "BPA": {
        "caixa":       ["1.01.01", "1.01.02"],
        "ativo_total": ["1"],
    },
    "BPP": {
        "divida_cp":      ["2.01.04"],
        "divida_lp":      ["2.02.01"],
        "patrimonio_liq": ["2.03"],
    },
}


# ─────────────────────────────────────────────
# 1. CARREGA DADOS CVM (uma vez, em memória)
# ─────────────────────────────────────────────

def carregar_cvm() -> dict[str, pd.DataFrame]:
    """
    Carrega os CSVs da CVM em memória.
    Prioriza os históricos (mais completos); usa ITRs por ano como backup.
    Filtra só ORDEM_EXERC = ÚLTIMO para evitar duplicatas do período comparativo.
    """
    dados = {}
    for tipo in ["DRE", "BPA", "BPP"]:
        frames = []

        # Histórico (cobre todos os trimestres de todos os anos)
        hist = DIR_CVM / f"cvm_historico_{tipo.lower()}_con.csv"
        if hist.exists():
            log.info(f"  Carregando histórico {tipo}...")
            df = pd.read_csv(hist, sep=";", encoding="latin1",
                             decimal=",", low_memory=False)
            frames.append(df)

        # ITRs por ano (backup / dados mais recentes)
        for ano in ANOS:
            itr = DIR_CVM / f"itr_cia_aberta_{tipo}_con_{ano}.csv"
            if itr.exists():
                log.info(f"  Carregando ITR {tipo} {ano}...")
                df = pd.read_csv(itr, sep=";", encoding="latin1",
                                 decimal=",", low_memory=False)
                frames.append(df)

        if not frames:
            log.warning(f"  Nenhum arquivo encontrado para {tipo}")
            continue

        df_concat = pd.concat(frames, ignore_index=True)

        # Remove duplicatas — mantém o último valor por empresa/data/conta
        # (substitui o filtro por ORDEM_EXERC que falha com encoding inconsistente)
        chave = ["DENOM_CIA", "DT_FIM_EXERC", "CD_CONTA"]
        chave_ok = [c for c in chave if c in df_concat.columns]
        df_concat = df_concat.drop_duplicates(
            subset=chave_ok, keep="last").reset_index(drop=True)

        # Converte valor para float
        df_concat["VL_CONTA"] = pd.to_numeric(
            df_concat["VL_CONTA"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce")

        dados[tipo] = df_concat
        log.info(f"  {tipo}: {len(df_concat):,} linhas após limpeza")

    return dados


# ─────────────────────────────────────────────
# 2. EXTRAÇÃO POR EMPRESA / TRIMESTRE
# ─────────────────────────────────────────────

def _extrair(df: pd.DataFrame, nome_cvm: str,
             data_alvo: str, codigos: list[str]) -> float:
    """
    Filtra o DataFrame por empresa, data e código de conta.
    Retorna o valor encontrado (em R$, já convertido de milhares).
    """
    col_nome = next((c for c in ["DENOM_CIA","NOME_CIA"]
                     if c in df.columns), None)
    if col_nome is None:
        return np.nan

    # Match exato de nome
    mask = df[col_nome].astype(str).str.strip() == nome_cvm.strip()
    sub = df[mask]
    if sub.empty:
        return np.nan

    # Filtra data
    sub = sub[sub["DT_FIM_EXERC"].astype(str).str.strip() == data_alvo]
    if sub.empty:
        return np.nan

    # Filtra contas
    sub = sub[sub["CD_CONTA"].astype(str).str.strip().isin(codigos)]
    if sub.empty:
        return np.nan

    val = sub["VL_CONTA"].dropna()
    if val.empty:
        return np.nan

    # CVM reporta em MIL — converte para R$ unidade
    escala = sub["ESCALA_MOEDA"].iloc[0] if "ESCALA_MOEDA" in sub.columns else "MIL"
    mult = 1000 if str(escala).strip().upper() in ["MIL","MILHARES"] else 1

    return float(val.iloc[-1]) * mult


def extrair_empresa_trimestre(
    dados: dict, nome_cvm: str, ano: int, trimestre: str
) -> dict:
    """Extrai todos os fundamentais de uma empresa em um trimestre."""
    data_alvo = f"{ano}{TRI_DATA[trimestre]}"  # ex: "2024-03-31"

    resultado = {}

    # DRE
    dre = dados.get("DRE")
    if dre is not None:
        for conta, codigos in CONTAS["DRE"].items():
            resultado[conta] = _extrair(dre, nome_cvm, data_alvo, codigos)

    # BPA
    bpa = dados.get("BPA")
    if bpa is not None:
        for conta, codigos in CONTAS["BPA"].items():
            resultado[conta] = _extrair(bpa, nome_cvm, data_alvo, codigos)

    # BPP
    bpp = dados.get("BPP")
    if bpp is not None:
        div_cp = _extrair(bpp, nome_cvm, data_alvo, CONTAS["BPP"]["divida_cp"])
        div_lp = _extrair(bpp, nome_cvm, data_alvo, CONTAS["BPP"]["divida_lp"])
        resultado["patrimonio_liq"] = _extrair(
            bpp, nome_cvm, data_alvo, CONTAS["BPP"]["patrimonio_liq"])

        # Dívida bruta = CP + LP
        if not np.isnan(div_cp) and not np.isnan(div_lp):
            resultado["divida_bruta"] = div_cp + div_lp
        elif not np.isnan(div_cp):
            resultado["divida_bruta"] = div_cp
        elif not np.isnan(div_lp):
            resultado["divida_bruta"] = div_lp
        else:
            resultado["divida_bruta"] = np.nan

    return resultado


# ─────────────────────────────────────────────
# 3. PREÇO HISTÓRICO — yfinance
# ─────────────────────────────────────────────

def carregar_precos(tickers: list[str]) -> pd.DataFrame:
    """
    Baixa preços históricos de todos os tickers de uma vez.
    Mais eficiente que baixar um por um.
    """
    log.info("\n  Baixando preços históricos (yfinance)...")
    tickers_sa = [t + ".SA" for t in tickers]
    hist = yf.download(
        tickers_sa,
        start="2022-12-01",
        end="2025-12-31",
        auto_adjust=True,
        progress=False,
    )
    return hist["Close"] if "Close" in hist.columns else hist


def preco_no_trimestre(precos: pd.DataFrame,
                       ticker: str, ano: int, trimestre: str) -> float:
    """Preço de fechamento no último dia útil do trimestre."""
    col = ticker + ".SA"
    if col not in precos.columns:
        return np.nan

    serie = precos[col].dropna()
    mes_fim = int(TRI_DATA[trimestre].split("-")[1])

    try:
        data_fim = pd.Timestamp(ano, mes_fim, 28)
        h = serie[serie.index <= data_fim]
        return float(h.iloc[-1]) if not h.empty else np.nan
    except Exception:
        return np.nan


def shares_outstanding(ticker: str) -> float:
    """Shares outstanding atual (yfinance). Aproximação para séries históricas."""
    try:
        info = yf.Ticker(ticker + ".SA").info
        return float(info.get("sharesOutstanding", np.nan))
    except Exception:
        return np.nan


# ─────────────────────────────────────────────
# 4. CÁLCULO DOS MÚLTIPLOS
# ─────────────────────────────────────────────

def calcular_multiplos(fund: dict, preco: float,
                       shares: float, div_yield: float) -> dict:
    """Calcula todos os múltiplos a partir dos fundamentais e do preço."""

    def sd(a, b):
        """Divisão segura — retorna NaN se inválido ou divisão por zero."""
        try:
            a, b = float(a), float(b)
            if not np.isfinite(a) or not np.isfinite(b) or b == 0:
                return np.nan
            return round(a / b, 4)
        except Exception:
            return np.nan

    lucro   = fund.get("lucro_liquido", np.nan)
    pl      = fund.get("patrimonio_liq", np.nan)
    divida  = fund.get("divida_bruta", np.nan)
    caixa   = fund.get("caixa", np.nan)
    ebit    = fund.get("ebit", np.nan)

    mkt_cap = (preco * shares) if (np.isfinite(preco) and
               np.isfinite(shares)) else np.nan
    div_liq = (divida - caixa) if (np.isfinite(divida) and
               np.isfinite(caixa)) else np.nan
    ev      = (mkt_cap + div_liq) if (np.isfinite(mkt_cap) and
               np.isfinite(div_liq)) else np.nan

    lpa = sd(lucro, shares)
    vpa = sd(pl,    shares)

    return {
        "pl_ratio":       sd(preco, lpa),
        "pvp":            sd(preco, vpa),
        "ev_ebitda":      sd(ev,    ebit),   # ebit como proxy de ebitda
        "dividend_yield": round(div_yield, 4) if np.isfinite(div_yield) else np.nan,
        "roe":            sd(lucro, pl),
        "divida_pl":      sd(div_liq, pl),
        # Dados brutos (úteis para debug e para o modelo de regressão)
        "lucro_liq_bi":   round(lucro / 1e9, 3) if np.isfinite(lucro) else np.nan,
        "pl_bi":          round(pl    / 1e9, 3) if np.isfinite(pl)    else np.nan,
        "divida_liq_bi":  round(div_liq / 1e9, 3) if np.isfinite(div_liq) else np.nan,
        "mkt_cap_bi":     round(mkt_cap / 1e9, 3) if np.isfinite(mkt_cap) else np.nan,
    }


# ─────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("COLETA DE FUNDAMENTAIS — CVM + yfinance")
    log.info(f"Empresas: {len(TICKERS)} | Período: 2023-2025 (1T/2T/3T)")
    log.info("=" * 60)

    # Carrega CVM uma vez
    log.info("\n[1/3] Carregando dados CVM...")
    dados_cvm = carregar_cvm()

    # Carrega preços uma vez (todos os tickers juntos)
    log.info("\n[2/3] Carregando preços e shares...")
    precos = carregar_precos(TICKERS)
    shares_map = {}
    div_yield_map = {}
    for ticker in TICKERS:
        log.info(f"  Shares {ticker}...")
        info = {}
        try:
            info = yf.Ticker(ticker + ".SA").info or {}
        except Exception:
            pass
        shares_map[ticker]    = float(info.get("sharesOutstanding", np.nan))
        div_yield_map[ticker] = float(info.get("dividendYield", np.nan))
        time.sleep(0.3)

    # Processa cada empresa × trimestre
    log.info("\n[3/3] Calculando múltiplos...")
    registros = []

    for ticker in TICKERS:
        nome_cvm = TICKER_NOME_CVM[ticker]
        shares   = shares_map[ticker]
        dy       = div_yield_map[ticker]
        log.info(f"\n  {ticker} | shares={shares:.0f}" if np.isfinite(shares)
                 else f"\n  {ticker} | shares=N/D")

        for ano in ANOS:
            for tri in TRIMESTRES_DISPONIVEIS:
                preco = preco_no_trimestre(precos, ticker, ano, tri)
                fund  = extrair_empresa_trimestre(dados_cvm, nome_cvm, ano, tri)
                mult  = calcular_multiplos(fund, preco, shares, dy)

                reg = {
                    "ticker":    ticker,
                    "ano":       ano,
                    "trimestre": tri,
                    "periodo":   f"{tri}{str(ano)[2:]}",
                    "preco":     round(preco, 2) if np.isfinite(preco) else np.nan,
                    **mult,
                }
                registros.append(reg)

                log.info(
                    f"    {ano} {tri} | P={reg['preco']} "
                    f"P/L={mult['pl_ratio']} "
                    f"P/VP={mult['pvp']} "
                    f"EV/EBITDA={mult['ev_ebitda']} "
                    f"ROE={mult['roe']}"
                )

    # Monta DataFrame
    df = pd.DataFrame(registros)
    df = df.sort_values(["ticker","ano","trimestre"]).reset_index(drop=True)

    # Winsorização 1%–99% (remove outliers extremos sem excluir linhas)
    cols_win = ["pl_ratio","pvp","ev_ebitda","roe","divida_pl"]
    for col in cols_win:
        if col in df.columns and df[col].notna().sum() > 10:
            p1  = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=p1, upper=p99)

    # Salva
    caminho = DIR_PROCESSED / "fatores_fundamentalistas.csv"
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    log.info(f"\nSalvo: {caminho}")
    log.info(f"Total: {len(df)} registros")

    # Resumo de cobertura
    log.info("\nCobertura por métrica (% não nulos):")
    for col in cols_win + ["preco"]:
        pct = df[col].notna().mean() * 100
        status = "✓" if pct > 70 else "⚠" if pct > 30 else "✗"
        log.info(f"  {status} {col:15}: {pct:.1f}%")

    # Preview
    print("\n" + "=" * 60)
    print("PREVIEW — primeiras 20 linhas")
    print("=" * 60)
    print(df[["ticker","periodo","preco","pl_ratio",
              "pvp","ev_ebitda","roe","divida_pl"]].head(20).to_string(index=False))

    return df


if __name__ == "__main__":
    df = main()