from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
PASTA_PROCESSED = BASE_DIR / "data" / "processed"
PASTA_YFINANCE = BASE_DIR / "data" / "raw" / "yfinance"

# Dicionário de Mapeamento CVM (CD_CVM de 6 dígitos) -> Ticker B3
# Adicione/Ajuste os tickers do IBrX-100 conforme a necessidade do seu universo
MAPA_CVM_TICKER = {
    "002437": "ELET3",
    "004170": "PETR4",
    "019348": "VALE3",
    "019356": "ITUB4",
    "001023": "BBDC4",
    "022470": "ABEV3",
    "014311": "BBAS3",
    "022870": "WEGE3",
    "019186": "RENT3",
    "020257": "PRIO3",
    "018376": "JBSS3",
}

def carregar_dados():
    # 1. Carregar CVM Limpo
    arq_cvm = PASTA_PROCESSED / "cvm_demonstrativos_limpos.csv"
    if not arq_cvm.exists():
        raise FileNotFoundError(f"Arquivo {arq_cvm} não encontrado. Execute limpar_cvm.py primeiro.")
    
    df_cvm = pd.read_csv(arq_cvm, dtype={"CD_CVM": str})
    df_cvm["CD_CVM"] = df_cvm["CD_CVM"].str.zfill(6)
    df_cvm["DT_REFER"] = pd.to_datetime(df_cvm["DT_REFER"])
    
    # Adicionar Ticker B3 via Mapeamento
    df_cvm["Ticker"] = df_cvm["CD_CVM"].map(MAPA_CVM_TICKER)
    df_cvm = df_cvm.dropna(subset=["Ticker"]).copy()

    # 2. Carregar Preços do yfinance
    arq_precos = PASTA_YFINANCE / "precos_historicos_yfinance.csv"
    if not arq_precos.exists():
        raise FileNotFoundError(f"Arquivo {arq_precos} não encontrado. Execute baixar_precos.py primeiro.")
        
    df_precos = pd.read_csv(arq_precos)
    df_precos["Date"] = pd.to_datetime(df_precos["Date"])
    
    return df_cvm, df_precos


def obter_preco_fim_trimestre(df_precos: pd.DataFrame) -> pd.DataFrame:
    """Extrai a última cotação disponível de cada trimestre para cada Ticker."""
    df_p = df_precos.sort_values(by=["Ticker", "Date"]).copy()
    
    # Identificar Ano e Trimestre (Ano-Tri)
    df_p["AnoTrimestre"] = df_p["Date"].dt.to_period("Q")
    
    # Pegar o último dia negociado de cada trimestre
    df_trimestral = df_p.groupby(["Ticker", "AnoTrimestre"]).last().reset_index()
    
    return df_trimestral[["Ticker", "AnoTrimestre", "Date", "Close"]].rename(
        columns={"Close": "Preco_Fechamento", "Date": "Data_Preco"}
    )


def calcular_multiplos():
    print("=" * 60)
    print("CALCULANDO MÚLTIPLOS DE VALUATION TRIMESTRAIS (P/L, P/VP, EV/EBITDA)")
    print("=" * 60)
    
    df_cvm, df_precos = carregar_dados()
    
    # Criar chave de período Ano-Trimestre no CVM
    df_cvm["AnoTrimestre"] = df_cvm["DT_REFER"].dt.to_period("Q")
    
    # Obter cotações do final de cada trimestre
    df_precos_trim = obter_preco_fim_trimestre(df_precos)
    
    # Merge entre CVM e Cotação
    df_merged = pd.merge(
        df_cvm,
        df_precos_trim,
        on=["Ticker", "AnoTrimestre"],
        how="inner"
    )
    
    # Cálculo dos Múltiplos (Ajustando escala quando necessário)
    # P/VP: Preço de Fechamento / (Patrimônio Líquido por ação estimado ou P/VP simplificado)
    # Se usarmos o Valor de Mercado total x Demonstrativo total:
    df_merged["P_L"] = np.where(df_merged["lucro_liquido"] > 0, df_merged["Preco_Fechamento"] / df_merged["lucro_liquido"], np.nan)
    df_merged["P_VP"] = np.where(df_merged["patrimonio_liquido"] > 0, df_merged["Preco_Fechamento"] / df_merged["patrimonio_liquido"], np.nan)
    df_merged["EV_EBIT"] = np.where(df_merged["ebit"] > 0, (df_merged["Preco_Fechamento"] + df_merged["divida_liquida"]) / df_merged["ebit"], np.nan)

    # Ordenar por Ticker e Data
    df_merged = df_merged.sort_values(by=["Ticker", "DT_REFER"]).reset_index(drop=True)
    
    saida = PASTA_PROCESSED / "multiplos_valuation_trimestrais.csv"
    df_merged.to_csv(saida, index=False)
    print(f"[ok] Múltiplos calculados e salvos em: {saida} ({len(df_merged):,} registros)")


if __name__ == "__main__":
    calcular_multiplos()