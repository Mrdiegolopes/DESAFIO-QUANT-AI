import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def extrair_tickers_b3_csv(file_path: str, output_json: str = "data/ibrx100_tickers.json") -> List[str]:
    path = Path(file_path)
    if not path.exists():
        logging.error(f"Arquivo não encontrado: {file_path}")
        return []

    logging.info(f"Lendo arquivo da B3: {file_path}")

    # Leitura adequada para o padrão de exportação da B3 (.csv / .xlsx)
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, skiprows=1, sep=";", encoding="latin1", index_col=False)
        except Exception:
            df = pd.read_csv(path, skiprows=1, sep=";", encoding="utf-8", index_col=False)
    else:
        try:
            df = pd.read_excel(path, skiprows=1)
        except Exception:
            df = pd.read_excel(path)

    # Identifica a coluna de Tickers/Códigos
    col_codigo = None
    for col in df.columns:
        col_str = str(col).strip().lower()
        if any(kw in col_str for kw in ["código", "codigo", "ticker", "ação", "acao"]):
            col_codigo = col
            break

    if not col_codigo:
        logging.error(f"Coluna de ticker não encontrada no arquivo. Colunas: {list(df.columns)}")
        return []

    # Limpeza e filtragem de tickers válidos
    raw_values = df[col_codigo].dropna().astype(str).str.strip().str.upper().tolist()

    # Ignora linhas de rodapé ("Quantidade Teórica Total", "Redutor")
    tickers = [
        t for t in raw_values 
        if len(t) in (5, 6) and not t.startswith("QUANT") and not t.startswith("REDUT")
    ]

    tickers = sorted(list(set(tickers)))

    payload: Dict[str, Any] = {
        "index": "IBRX100",
        "source": path.name,
        "total_count": len(tickers),
        "tickers": tickers
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logging.info(f"Sucesso! {len(tickers)} tickers extraídos e salvos em: {output_path}")
    return tickers


if __name__ == "__main__":
    # Caminho do arquivo enviado
    ARQUIVO_B3 = r"C:\Users\55859\value-factor-model-ibrx100\data\raw\b3\IBXXDia_05-08-26.xlsx"
    
    lista_tickers = extrair_tickers_b3_csv(file_path=ARQUIVO_B3)