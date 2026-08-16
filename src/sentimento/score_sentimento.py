"""
Score de Sentimento via LLM — Earnings Calls IBrX-100

Para cada texto extraído, chama o LLM (Anthropic ou Gemini) e extrai:
  - score de sentimento: float entre -1.0 (muito negativo) e +1.0 (muito positivo)
  - tom: muito_positivo | positivo | neutro | cauteloso | negativo
  - resumo: 1-2 frases do tom geral
  - evidencias: até 3 citações literais que justificam o score

Saída: data/processed/scores_sentimento.csv

Princípio: o LLM lê o texto e classifica — cada classificação carrega
citação literal verificável (citation tracking feito no módulo seguinte).
"""

import os
import json
import time
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import Optional


from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ       = Path(__file__).parent.parent.parent
DIR_TEXTOS = RAIZ / "data" / "processed" / "textos_earnings"
CSV_INDEX  = RAIZ / "data" / "processed" / "textos_earnings.csv"
CSV_SAIDA  = RAIZ / "data" / "processed" / "scores_sentimento.csv"

MAX_CHARS  = 12000   # trunca textos muito longos para economizar tokens
MAX_OUTPUT = 1000


# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

class Tom(str, Enum):
    MUITO_POSITIVO = "muito_positivo"
    POSITIVO       = "positivo"
    NEUTRO         = "neutro"
    CAUTELOSO      = "cauteloso"
    NEGATIVO       = "negativo"

TOM_SCORE = {
    "muito_positivo":  1.0,
    "positivo":        0.5,
    "neutro":          0.0,
    "cauteloso":      -0.5,
    "negativo":       -1.0,
}

class Evidencia(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citacao: str = Field(..., description="Trecho literal do texto.")
    contexto: str = Field(..., description="Por que esse trecho justifica o tom.")

class ScoreSentimento(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tom: Tom
    resumo: str = Field(..., description="1-2 frases sobre o tom geral.")
    evidencias: list[Evidencia] = Field(..., min_length=1, max_length=3)


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

PROMPT_SISTEMA = """Você é um analista sênior de Equity Research especializado em
análise de earnings calls de empresas brasileiras. Analise o texto fornecido e
classifique o tom geral do management.

REGRAS:
1. Classifique o tom como: muito_positivo, positivo, neutro, cauteloso ou negativo
2. Baseie-se APENAS no texto fornecido — não use conhecimento externo
3. Cada evidência deve ser uma citação LITERAL do texto (copiada palavra por palavra)
4. Se o texto for um release de resultados (não transcrição), foque na linguagem
   dos comentários do management sobre desempenho e perspectivas
5. Seja objetivo — ignore linguagem corporativa genérica e foque em sinais reais
   de confiança ou preocupação do management"""

PROMPT_USUARIO = """Analise o tom do management neste documento de earnings call/release:

EMPRESA: {ticker} | PERÍODO: {periodo}

TEXTO:
{texto}

Classifique o tom e forneça evidências literais."""


# ─────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────

def _provedor() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    raise RuntimeError(
        "Defina ANTHROPIC_API_KEY ou GEMINI_API_KEY.\n"
        "PowerShell: $env:ANTHROPIC_API_KEY='sk-ant-...'"
    )


def _chamar_anthropic(texto_usuario: str) -> ScoreSentimento:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=MAX_OUTPUT,
        system=PROMPT_SISTEMA,
        tools=[{
            "name": "classificar_sentimento",
            "description": "Classifica o sentimento do earnings call.",
            "input_schema": ScoreSentimento.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "classificar_sentimento"},
        messages=[{"role": "user", "content": texto_usuario}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return ScoreSentimento.model_validate(block.input)
    raise RuntimeError("Anthropic não retornou tool_use")


def _resolver_refs(schema: dict) -> dict:
    """Achata $ref/$defs para o Gemini (mesmo fix do Case 2)."""
    defs = schema.pop("$defs", {})
    def _resolve(node):
        if isinstance(node, dict):
            node = {k: v for k, v in node.items()
                    if k != "additionalProperties"}
            if "$ref" in node:
                ref = node["$ref"].split("/")[-1]
                resolved = dict(defs.get(ref, {}))
                resolved.pop("title", None)
                resolved.pop("additionalProperties", None)
                return _resolve(resolved)
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(i) for i in node]
        return node
    return _resolve(schema)


def _chamar_gemini(texto_usuario: str) -> ScoreSentimento:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    schema = _resolver_refs(ScoreSentimento.model_json_schema())
    cfg = types.GenerateContentConfig(
        system_instruction=PROMPT_SISTEMA,
        temperature=0,
        response_mime_type="application/json",
        response_schema=schema,
        max_output_tokens=MAX_OUTPUT,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=texto_usuario,
        config=cfg,
    )
    return ScoreSentimento.model_validate(json.loads(resp.text))


def classificar_sentimento(ticker: str, periodo: str,
                            texto: str) -> Optional[ScoreSentimento]:
    """Chama o LLM e retorna o score de sentimento."""
    # Trunca texto muito longo
    texto_truncado = texto[:MAX_CHARS] if len(texto) > MAX_CHARS else texto
    prompt_user = PROMPT_USUARIO.format(
        ticker=ticker, periodo=periodo, texto=texto_truncado)

    prov = _provedor()
    for tentativa in range(2):
        try:
            if prov == "anthropic":
                return _chamar_anthropic(prompt_user)
            else:
                return _chamar_gemini(prompt_user)
        except Exception as e:
            if tentativa == 0:
                log.warning(f"    Tentativa 1 falhou: {e}. Repetindo...")
                time.sleep(3)
            else:
                log.error(f"    Falha definitiva: {e}")
                return None
    return None


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("SCORE DE SENTIMENTO — LLM")
    log.info("=" * 60)

    # Carrega índice de textos
    if not CSV_INDEX.exists():
        log.error(f"Índice não encontrado: {CSV_INDEX}")
        log.error("Rode src/sentimento/extrair_texto.py primeiro.")
        return

    df_idx = pd.read_csv(CSV_INDEX, encoding="utf-8-sig")
    log.info(f"Textos disponíveis: {len(df_idx)}")

    # Carrega scores já processados (para retomar de onde parou)
    if CSV_SAIDA.exists():
        df_existente = pd.read_csv(CSV_SAIDA, encoding="utf-8-sig")
        ja_processados = set(
            zip(df_existente["ticker"], df_existente["ano"].astype(str),
                df_existente["trimestre"]))
        log.info(f"Já processados: {len(df_existente)} — continuando do resto")
    else:
        df_existente = pd.DataFrame()
        ja_processados = set()

    registros = []
    sucesso = 0
    falha   = 0

    for _, row in df_idx.iterrows():
        ticker    = str(row["ticker"])
        ano       = int(row["ano"])
        trimestre = str(row["trimestre"])
        periodo   = str(row["periodo"])

        # Pula se já processado
        if (ticker, str(ano), trimestre) in ja_processados:
            log.info(f"  {ticker} {periodo} — já processado, pulando")
            continue

        # Lê texto
        caminho_txt = Path(row["caminho_txt"])
        if not caminho_txt.exists():
            log.warning(f"  {ticker} {periodo} — arquivo não encontrado")
            falha += 1
            continue

        texto = caminho_txt.read_text(encoding="utf-8")
        log.info(f"  {ticker} {periodo} ({len(texto.split())} palavras)...")

        # Chama LLM
        score = classificar_sentimento(ticker, periodo, texto)

        if score is None:
            falha += 1
            continue

        # Score numérico
        score_num = TOM_SCORE.get(score.tom.value, 0.0)

        reg = {
            "ticker":     ticker,
            "ano":        ano,
            "trimestre":  trimestre,
            "periodo":    periodo,
            "tom":        score.tom.value,
            "score":      score_num,
            "resumo":     score.resumo,
            "evidencia_1": score.evidencias[0].citacao if len(score.evidencias) > 0 else "",
            "evidencia_2": score.evidencias[1].citacao if len(score.evidencias) > 1 else "",
            "evidencia_3": score.evidencias[2].citacao if len(score.evidencias) > 2 else "",
        }
        registros.append(reg)
        sucesso += 1

        log.info(f"    ✓ tom={score.tom.value} | score={score_num} | {score.resumo[:60]}...")

        # Salva incrementalmente a cada 5 processados
        if sucesso % 5 == 0:
            _salvar(registros, df_existente)

        time.sleep(0.5)  # rate limit

    # Salva final
    _salvar(registros, df_existente)

    log.info(f"\n{'='*60}")
    log.info(f"Sucesso: {sucesso} | Falha: {falha}")
    log.info(f"CSV salvo: {CSV_SAIDA}")

    # Preview
    df_final = pd.read_csv(CSV_SAIDA, encoding="utf-8-sig")
    print(f"\nDistribuição de tons:")
    print(df_final["tom"].value_counts().to_string())
    print(f"\nScore médio por ticker:")
    print(df_final.groupby("ticker")["score"].mean().round(3).to_string())


def _salvar(registros_novos: list, df_existente: pd.DataFrame):
    df_novo = pd.DataFrame(registros_novos)
    df_final = pd.concat([df_existente, df_novo], ignore_index=True) \
        if not df_existente.empty else df_novo
    df_final = df_final.drop_duplicates(
        subset=["ticker","ano","trimestre"], keep="last")
    df_final.to_csv(CSV_SAIDA, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()