"""
Citation Tracking — CARCARÁ
Verifica deterministicamente se as citações extraídas pelo LLM
existem nos textos-fonte das earnings calls.

Lê: data/processed/scores_sentimento.csv (citações do LLM)
     data/processed/textos_earnings/TICKER_ANO_TRI.txt (textos-fonte)

Saída: data/processed/auditoria_citacoes.csv
       outputs/grounding_resumo.txt

Metodologia (herdada do Case 1 do BBI, com 3 correções documentadas):
  1. Match exato por substring após normalização
  2. Sem pontuação de fechamento (LLM fecha citação com ponto que
     não existe naquele ponto exato do texto-fonte)
  3. SequenceMatcher com janela sem teto fixo (correção do bug de
     janela limitada a 600 chars que rejeitava citações compostas)
"""

import re
import logging
import unicodedata
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ        = Path(__file__).parent.parent.parent
DIR_TEXTOS  = RAIZ / "data" / "processed" / "textos_earnings"
CSV_SENT    = RAIZ / "data" / "processed" / "scores_sentimento.csv"
CSV_SAIDA   = RAIZ / "data" / "processed" / "auditoria_citacoes.csv"
TXT_RESUMO  = RAIZ / "outputs" / "grounding_resumo.txt"

SIMILARIDADE_MINIMA = 0.85
RE_RETICENCIAS = re.compile(r"\s*(?:\.{3,}|…)\s*")


# ─────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────

class StatusCitacao(str, Enum):
    encontrada    = "encontrada"
    aproximada    = "aproximada"
    nao_encontrada = "nao_encontrada"
    vazia         = "vazia"


# ─────────────────────────────────────────────
# NORMALIZAÇÃO
# ─────────────────────────────────────────────

def _norm(s: str) -> str:
    """
    Normalização robusta — remove acentos, unifica aspas, remove
    inserções editoriais entre colchetes e pontuação de fechamento.
    (Correção 2 do Case 1: diferenças tipográficas geravam falsos negativos)
    """
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    # Remove inserções editoriais [...]
    s = re.sub(r"\[[^\]]*\]", "", s)
    # Unifica aspas
    s = re.sub(r"[\"'\u2018\u2019\u201c\u201d]", "", s)
    s = s.replace("–", "-").replace("—", "-").replace("…", "...")
    # Colapsa espaços
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ─────────────────────────────────────────────
# JANELAS (sem teto fixo — Correção 1 do Case 1)
# ─────────────────────────────────────────────

def _janelas(texto: str, tamanho_citacao: int):
    """
    Gera janelas do texto-fonte sem teto fixo de tamanho.
    (Correção 1: teto de 600 chars rejeitava citações longas legítimas)
    """
    norm = _norm(texto)
    tamanho = max(40, tamanho_citacao + 40)
    step = max(20, tamanho // 3)
    for i in range(0, max(1, len(norm) - tamanho + 1), step):
        yield norm[i:i + tamanho]
    if len(norm) > tamanho:
        yield norm[-tamanho:]


# ─────────────────────────────────────────────
# VERIFICAÇÃO
# ─────────────────────────────────────────────

def _melhor_similaridade(citacao_norm: str,
                          texto_fonte: str) -> tuple[float, str | None]:
    """
    Busca a melhor janela do texto-fonte para um trecho contíguo.
    Inclui tentativa sem pontuação de fechamento.
    (Correção 3 do Case 1: ponto final adicionado pelo LLM causava miss)
    """
    if not citacao_norm:
        return 0.0, None
    fonte_norm = _norm(texto_fonte)

    # 1. Match exato
    if citacao_norm in fonte_norm:
        return 1.0, citacao_norm

    # 2. Sem pontuação de fechamento
    sem_pont = citacao_norm.rstrip(".,;: ")
    if sem_pont and sem_pont != citacao_norm and sem_pont in fonte_norm:
        return 1.0, sem_pont

    # 3. SequenceMatcher com janela sem teto
    melhor = (0.0, None)
    for janela in _janelas(texto_fonte, len(citacao_norm)):
        score = SequenceMatcher(None, citacao_norm, janela).ratio()
        if score > melhor[0]:
            melhor = (score, janela)
            if melhor[0] >= 0.999:
                break
    return melhor


def verificar_citacao(citacao: str, texto_fonte: str) -> dict:
    """
    Verifica uma citação contra o texto-fonte.
    Citações com reticências são verificadas segmento por segmento
    e o pior score entre os segmentos determina o status final.
    """
    c = _norm(citacao)

    if not c or len(c.split()) < 2:
        return {"status": StatusCitacao.vazia, "similaridade": 0.0}

    segmentos = [s.strip() for s in RE_RETICENCIAS.split(citacao) if s.strip()]

    if len(segmentos) <= 1:
        score, _ = _melhor_similaridade(c, texto_fonte)
    else:
        resultados_seg = []
        for seg in segmentos:
            seg_norm = _norm(seg)
            if len(seg_norm.split()) < 2:
                continue
            s, _ = _melhor_similaridade(seg_norm, texto_fonte)
            resultados_seg.append(s)
        score = min(resultados_seg) if resultados_seg else 0.0

    if score >= 0.999:
        status = StatusCitacao.encontrada
    elif score >= SIMILARIDADE_MINIMA:
        status = StatusCitacao.aproximada
    else:
        status = StatusCitacao.nao_encontrada

    return {"status": status, "similaridade": round(score, 3)}


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("CITATION TRACKING — CARCARÁ")
    log.info("=" * 60)

    if not CSV_SENT.exists():
        log.error(f"Não encontrado: {CSV_SENT}")
        return

    df_sent = pd.read_csv(CSV_SENT, encoding="utf-8-sig")
    log.info(f"Registros de sentimento: {len(df_sent)}")

    # Colunas de evidência
    cols_ev = ["evidencia_1", "evidencia_2", "evidencia_3"]
    cols_ev = [c for c in cols_ev if c in df_sent.columns]
    log.info(f"Colunas de evidência: {cols_ev}")

    registros = []
    total = 0
    encontradas = 0
    aproximadas = 0
    nao_encontradas = 0

    for _, row in df_sent.iterrows():
        ticker    = str(row["ticker"])
        ano       = int(row["ano"])
        trimestre = str(row["trimestre"])

        # Lê texto-fonte
        nome_txt = f"{ticker}_{ano}_{trimestre}.txt"
        caminho_txt = DIR_TEXTOS / nome_txt
        if not caminho_txt.exists():
            log.warning(f"  Texto não encontrado: {nome_txt}")
            continue

        texto = caminho_txt.read_text(encoding="utf-8")

        for col in cols_ev:
            citacao = str(row.get(col, "")).strip()
            if not citacao or citacao == "nan":
                continue

            resultado = verificar_citacao(citacao, texto)
            total += 1

            if resultado["status"] == StatusCitacao.encontrada:
                encontradas += 1
            elif resultado["status"] == StatusCitacao.aproximada:
                aproximadas += 1
            else:
                nao_encontradas += 1

            registros.append({
                "ticker":       ticker,
                "ano":          ano,
                "trimestre":    trimestre,
                "periodo":      f"{trimestre}{str(ano)[2:]}",
                "coluna":       col,
                "citacao":      citacao[:100],
                "status":       resultado["status"],
                "similaridade": resultado["similaridade"],
            })

    # Salva
    df_audit = pd.DataFrame(registros)
    df_audit.to_csv(CSV_SAIDA, index=False, encoding="utf-8-sig")

    # Métricas
    taxa_grounding = (encontradas + aproximadas) / total if total > 0 else 0.0

    resumo = f"""CITATION TRACKING — CARCARÁ
{'='*50}
Total de citações verificadas: {total}
  Encontradas (match exato):   {encontradas} ({100*encontradas/total:.1f}%)
  Aproximadas (≥85% similar):  {aproximadas} ({100*aproximadas/total:.1f}%)
  Não encontradas:             {nao_encontradas} ({100*nao_encontradas/total:.1f}%)

Taxa de grounding: {taxa_grounding:.1%}

Metodologia:
  - Normalização robusta (acentos, aspas, colchetes editoriais)
  - Verificação sem teto de janela (corrige rejeição de citações longas)
  - Citações compostas (com reticências) verificadas por segmento
  - Pior segmento determina o status final (preserva proteção anti-alucinação)
  - Limiar de similaridade: {SIMILARIDADE_MINIMA:.0%}

Por ticker:
"""

    if not df_audit.empty:
        por_ticker = df_audit.groupby("ticker").apply(
            lambda g: (g["status"].isin(
                [StatusCitacao.encontrada, StatusCitacao.aproximada]
            )).mean()
        ).round(3)
        for t, v in por_ticker.items():
            resumo += f"  {t:<8}: {v:.1%}\n"

    log.info(f"\n{resumo}")
    (TXT_RESUMO).write_text(resumo, encoding="utf-8")
    log.info(f"Auditoria salva: {CSV_SAIDA}")
    log.info(f"Resumo salvo: {TXT_RESUMO}")
    log.info(f"\nTaxa de grounding final: {taxa_grounding:.1%}")

    return df_audit


if __name__ == "__main__":
    main()