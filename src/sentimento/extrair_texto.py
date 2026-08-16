"""
Extração de texto dos PDFs de earnings calls.
Lê todos os PDFs em pdfs_coletados/TICKER/ANO/TRIMESTRE/
e salva o texto limpo em data/processed/textos_earnings/TICKER_ANO_TRI.txt

Saída principal: data/processed/textos_earnings.csv
  ticker | ano | trimestre | periodo | n_palavras | caminho_txt | texto
"""

import re
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ        = Path(__file__).parent.parent.parent
DIR_PDFS    = RAIZ / "pdfs_coletados"
DIR_TEXTOS  = RAIZ / "data" / "processed" / "textos_earnings"
DIR_TEXTOS.mkdir(parents=True, exist_ok=True)
CSV_SAIDA   = RAIZ / "data" / "processed" / "textos_earnings.csv"

# Mínimo de palavras para considerar extração bem-sucedida
MIN_PALAVRAS = 100


def _limpar_texto(texto: str) -> str:
    """Remove ruído típico de PDFs financeiros brasileiros."""
    # Remove headers/footers repetitivos
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    # Remove números de página isolados
    texto = re.sub(r'^\s*\d+\s*$', '', texto, flags=re.MULTILINE)
    # Remove linhas só com traços/underscores
    texto = re.sub(r'^[-_=]{3,}\s*$', '', texto, flags=re.MULTILINE)
    # Colapsa espaços múltiplos
    texto = re.sub(r' {2,}', ' ', texto)
    # Remove caracteres de controle
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', texto)
    return texto.strip()


def extrair_texto_pdf(caminho: Path) -> str | None:
    """
    Extrai texto de um PDF. Tenta pdfplumber primeiro (melhor para
    PDFs com tabelas e formatação complexa), depois pypdf como fallback.
    """
    # Tenta pdfplumber
    try:
        import pdfplumber
        texto = ""
        with pdfplumber.open(caminho) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texto += t + "\n"
        if len(texto.split()) >= MIN_PALAVRAS:
            return _limpar_texto(texto)
    except Exception as e:
        log.debug(f"pdfplumber falhou em {caminho.name}: {e}")

    # Fallback: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(caminho))
        texto = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                texto += t + "\n"
        if len(texto.split()) >= MIN_PALAVRAS:
            return _limpar_texto(texto)
    except Exception as e:
        log.debug(f"pypdf falhou em {caminho.name}: {e}")

    return None


def processar_todos() -> pd.DataFrame:
    """
    Percorre a estrutura pdfs_coletados/TICKER/ANO/TRIMESTRE/
    e extrai texto de cada PDF encontrado.
    """
    registros = []
    pdfs = sorted(DIR_PDFS.rglob("*.pdf"))
    log.info(f"Total de PDFs encontrados: {len(pdfs)}")

    sucesso = 0
    falha   = 0

    for caminho in pdfs:
        # Infere ticker/ano/trimestre da estrutura de pastas
        partes = caminho.relative_to(DIR_PDFS).parts
        if len(partes) < 4:
            continue

        ticker    = partes[0].upper()
        try:
            ano   = int(partes[1])
        except ValueError:
            continue
        trimestre = partes[2].upper()
        periodo   = f"{trimestre}{str(ano)[2:]}"

        log.info(f"  {ticker} {periodo} — {caminho.name[:50]}")

        texto = extrair_texto_pdf(caminho)

        if texto is None or len(texto.split()) < MIN_PALAVRAS:
            log.warning(f"    ✗ Extração falhou ou texto insuficiente")
            falha += 1
            continue

        # Salva .txt individual
        nome_txt = f"{ticker}_{ano}_{trimestre}.txt"
        caminho_txt = DIR_TEXTOS / nome_txt
        caminho_txt.write_text(texto, encoding="utf-8")

        n_palavras = len(texto.split())
        log.info(f"    ✓ {n_palavras} palavras")
        sucesso += 1

        registros.append({
            "ticker":      ticker,
            "ano":         ano,
            "trimestre":   trimestre,
            "periodo":     periodo,
            "n_palavras":  n_palavras,
            "caminho_txt": str(caminho_txt),
            "texto":       texto,
        })

    df = pd.DataFrame(registros)
    if not df.empty:
        # Salva CSV (sem coluna texto para não ficar gigante)
        df_csv = df.drop(columns=["texto"])
        df_csv.to_csv(CSV_SAIDA, index=False, encoding="utf-8-sig")
        log.info(f"\nCSV salvo: {CSV_SAIDA}")

    log.info(f"\nResumo: {sucesso} extraídos | {falha} falhas")
    log.info(f"Cobertura: {sucesso}/{sucesso+falha} ({100*sucesso/(sucesso+falha):.1f}%)")

    return df


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("EXTRAÇÃO DE TEXTO — PDFs DE EARNINGS CALLS")
    log.info("=" * 60)

    # Verifica dependências
    try:
        import pdfplumber
        log.info("pdfplumber: OK")
    except ImportError:
        log.warning("pdfplumber não instalado — pip install pdfplumber")

    try:
        from pypdf import PdfReader
        log.info("pypdf: OK")
    except ImportError:
        log.warning("pypdf não instalado — pip install pypdf")

    df = processar_todos()
    if not df.empty:
        print(f"\nTextos extraídos: {len(df)}")
        print(df[["ticker","periodo","n_palavras"]].to_string(index=False))