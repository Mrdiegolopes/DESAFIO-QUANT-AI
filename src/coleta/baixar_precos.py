from pathlib import Path 
import pandas as pd
import yfinance as yf 


ARQUIVO_IBRX = Path ("data/raw/b3/IBXXDia_05-08-26.xlsx")
PASTA_SAIDA = Path ("data/raw/yfinance")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

DATA_INICIO = "2015-01-01"
DATA_FIM = None



df = pd.read_excel(ARQUIVO_IBRX,header=1)
TICKER_COL = "Código"
if TICKER_COL not in df.columns:
    raise ValueError(
        f"Coluna '{TICKER_COL}' não encontrada.\n"
        f"Colunas disponíveis: {df.columns.tolist()}"
    )



tickers = (
    df[TICKER_COL]
    .dropna()
    .astype(str)
    .str.strip()

)

tickers = tickers[
    tickers.str.match(r"^[A-Z]{4}\d{1,2}$")
].unique()

tickers_yahoo = [f"{ticker}.SA" for ticker in tickers]

print(f"{len(tickers_yahoo)} empresas encontradas.")

print("COLETA DE PREÇOS - YAHOO FINANCE")
print(f"Empresas encontradas: {len(tickers_yahoo)}")


#baixando 

dados = yf.download(
    tickers=tickers_yahoo,
    start=DATA_INICIO,
    end=DATA_FIM,
    auto_adjust=True,
    group_by="ticker",
    progress=True,
)

# Transforma o MultiIndex em formato longo
dados = (
    dados
    .stack(level=0)
    .reset_index()
    .rename(columns={"level_1": "Ticker"})
)

# Remove o .SA
dados["Ticker"] = dados["Ticker"].str.replace(".SA", "", regex=False)

# Ordena
dados = dados.sort_values(
    by=["Ticker", "Date"]
).reset_index(drop=True)

# Reorganiza as colunas
dados = dados[
    [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
]


# Salvando os dados em arquivos CSV separados para cada empresa 

arquivo_saida = PASTA_SAIDA / "precos_historicos_yfinance.csv"
dados.to_csv(arquivo_saida, index=True)

print(dados.head())
print(dados.columns)

print(f"dados.head():\n{dados.head()}")
print(f"\nArquivo salvo em:\n{arquivo_saida}")