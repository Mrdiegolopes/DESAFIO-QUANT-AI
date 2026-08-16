# Fortal Gradient — Quality-Adjusted Value Factor Model with LLM Sentiment

> *Precisão de rapina. Inteligência de mercado.*

**Desafio Quant AI Itaú Asset 2026** — IBrX-100 Portfolio Optimization via Quality-Adjusted Value Factors & Verified Earnings Call Sentiment + Black-Litterman

---

## Sumário

- [Visão Geral](#visão-geral)
- [Estratégia](#estratégia)
- [Arquitetura](#arquitetura)
- [Resultados](#resultados)
- [Instalação](#instalação)
- [Como Rodar](#como-rodar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Dados](#dados)
- [Metodologia](#metodologia)
- [Limitações](#limitações)
- [Referências](#referências)

---

## Visão Geral

O CARCARÁ é um modelo quantitativo que combina três fontes de sinal para ranquear ações do IBrX-100 por retorno esperado:

1. **Fatores de Valuation e Quality** (P/VP, ROE, EV/EBITDA, Dívida/PL) extraídos dos demonstrativos financeiros da CVM
2. **Sentimento de Earnings Calls** extraído via LLM (Anthropic/Gemini) com verificação determinística de citações (citation tracking)
3. **Otimização Black-Litterman** que combina as views da regressão com o prior de equilíbrio do IBrX-100

**Pergunta central e testável:** sentimento do management nas earnings calls tem poder preditivo sobre o retorno futuro além do que os fatores de valuation já explicam?

**Resposta empírica:** sim — o sentimento normalizado é o único fator estatisticamente significativo na amostra (β=0.033, p=0.033), sugerindo que o LLM captura informação prospectiva que os múltiplos contábeis não capturam.

---

## Estratégia

### Equação do Modelo

```
Retorno(t+1) = β₁×P/VP(t) + β₂×EV/EBITDA(t) + β₃×P/L(t)
             + β₄×ROE(t) + β₅×Dívida/PL(t) + β₆×Sentimento_norm(t) + ε
```

- **Estimação:** OLS com erros HAC (Newey-West, 2 lags) — corrige heterocedasticidade e autocorrelação em séries trimestrais
- **Normalização:** Z-score cross-sectional por trimestre antes da regressão
- **Winsorização:** 1%-99% por trimestre para remover outliers extremos

### Por que Value + Quality + Sentimento

| Sinal | Problema isolado | Como o outro resolve |
|---|---|---|
| Só Value | Value traps — empresa barata porque deteriorando | Quality filtra: barato + ROE alto = oportunidade real |
| Só Quality | Empresas boas mas caras não entregam retorno extra | Value filtra: só compra qualidade com desconto |
| Só Sentimento | LLM pode ser confiante sobre empresa cara ou em deterioração | Value + Quality ancoram o sinal qualitativo |

### Diferencial vs. Baseline (Lee et al., 2025)

| Dimensão | Lee et al. (2025) | CARCARÁ |
|---|---|---|
| Fonte das views BL | LLM prevendo retorno diretamente | Regressão OLS+HAC em fatores históricos |
| Grounding | Nenhum — LLM "chuta" | Citation tracking determinístico (84.9%) |
| Incerteza das views | Variância de 100 repetições do LLM | R² e p-valores da regressão |
| Mercado | S&P 500 | IBrX-100 (gap na literatura brasileira) |
| Sentimento | Não usa | Fator formal testado na regressão |

---

## Arquitetura

```
[CVM - ITR/DFP]          [Earnings Calls - RI das empresas]
      │                              │
      ▼                              ▼
[Fundamentais trimestrais]    [Extração de texto (PDF)]
P/L · P/VP · EV/EBITDA           pdfplumber + pypdf
ROE · Dívida/PL                       │
      │                         [LLM Sentimento]
      │                    Anthropic/Gemini → score [-1,+1]
      │                         + Citation Tracking (84.9%)
      │                              │
      └──────────────┬───────────────┘
                     ▼
           [Regressão OLS + HAC]
           β₁...β₆ com Newey-West
           R²=6.4% | Sent. p=0.033
                     │
                     ▼
           [Black-Litterman]
           Views (rank percentil) + Prior IBrX-100
           Pesos ótimos com restrições
                     │
                     ▼
           [Backtest Out-of-Sample]
           Janela expansível | Spearman | Alpha
```

---

## Resultados

### Motor de Fatores (OLS+HAC Pooled)

| Fator | Beta | t-stat | p-valor | Sig |
|---|---|---|---|---|
| P/L | 0.0114 | 0.454 | 0.650 | |
| P/VP | -0.0285 | -1.483 | 0.138 | |
| EV/EBITDA | -0.0085 | -0.310 | 0.757 | |
| ROE | 0.0233 | 0.624 | 0.533 | |
| Dívida/PL | -0.0171 | -0.460 | 0.645 | |
| **Sentimento** | **0.0330** | **2.129** | **0.033** | **\*\*** |

- **R²:** 6.4% | **R² ajustado:** 0.55% | **N:** 103 obs | **Durbin-Watson:** 2.54

### Backtest Out-of-Sample (Janela Expansível)

| Métrica | Valor |
|---|---|
| Spearman full-sample (inflacionado) | 0.264 |
| Spearman OOS médio | 0.183 |
| Degradação (inflação de métrica) | 0.081 |
| **Retorno BL médio por trimestre** | **7.49% (+29.9%/ano)** |
| **Alpha BL vs Equal Weight** | **+1.78%/trimestre (+7.1%/ano)** |
| **Alpha Top 5 vs Equal Weight** | **+1.71%/trimestre (+6.8%/ano)** |
| BL bateu Equal Weight | 5 de 7 períodos |
| Top 5 bateu Equal Weight | 5 de 7 períodos |
| Períodos OOS testados | 7 |

> **Nota metodológica:** as views do Black-Litterman são construídas a partir de betas estimados sem z-score (escala de retorno real), o que elimina o problema de escala que afetava a versão anterior e resulta em alpha positivo do portfólio BL.

**Por período:**

| Período | Spearman OOS | BL | Top 5 | Equal Weight | Alpha BL | Alpha Top 5 |
|---|---|---|---|---|---|---|
| 3T23 | 0.191 | +19.0% | +19.9% | +19.5% | -0.5% | +0.5% |
| 1T24 | 0.354 | +0.8% | -0.8% | -5.2% | +6.0% | +4.4% |
| 2T24 | 0.015 | +10.4% | +8.2% | +11.1% | -0.7% | -2.9% |
| 3T24 | 0.191 | -9.1% | -7.2% | -10.9% | +1.7% | +3.7% |
| 1T25 | 0.301 | +9.6% | +11.4% | +5.5% | +4.1% | +5.8% |
| 2T25 | 0.288 | +9.5% | +8.5% | +7.5% | +1.9% | +0.9% |
| 3T25 | -0.059 | +12.4% | +12.0% | +12.4% | -0.0% | -0.5% |

### Citation Tracking

| Métrica | Valor |
|---|---|
| Total de citações verificadas | 498 |
| Encontradas (match exato) | 404 (81.1%) |
| Aproximadas (≥85% similar) | 19 (3.8%) |
| **Taxa de grounding** | **84.9%** |

---

## Instalação

```bash
git clone https://github.com/Mrdiegolopes/DESAFIO-QUANT-AI
cd DESAFIO-QUANT-AI

pip install -r requirements.txt
```

**Dependências principais:**
```
pandas numpy scipy statsmodels yfinance
pdfplumber pypdf anthropic google-genai
pydantic python-dotenv
```

**Variáveis de ambiente** (crie um `.env` na raiz):
```
ANTHROPIC_API_KEY=sk-ant-...   # ou
GEMINI_API_KEY=...             # Gemini como alternativa
```

---

## Como Rodar

Execute os módulos na seguinte ordem:

```bash
# 1. Coleta de fundamentais (CVM + yfinance)
python src/coleta/coletar_fundamentais.py

# 2. Extração de texto dos PDFs
python src/sentimento/extrair_texto.py

# 3. Score de sentimento via LLM
python src/sentimento/score_sentimento.py

# 4. Citation tracking (verificação de citações)
python src/sentimento/validar_citacoes.py

# 5. Merge dos datasets
python src/processamento/merge_datasets.py

# 6. Motor de fatores (OLS+HAC)
python src/fatores/factor_builder.py

# 7. Black-Litterman
python src/modelos/black_litterman.py

# 8. Backtest out-of-sample
python src/validacao/backtest.py
```

---

## Estrutura do Projeto

```
DESAFIO-QUANT-AI/
├── .env                          # Chaves de API (não versionado)
├── .gitignore
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── cvm/                  # ITRs e DFPs da CVM
│   │   │   ├── itr_cia_aberta_DRE_con_2023.csv
│   │   │   ├── itr_cia_aberta_BPA_con_2023.csv
│   │   │   ├── itr_cia_aberta_BPP_con_2023.csv
│   │   │   └── ... (2024, 2025)
│   │   └── b3/
│   │       └── IBXXDia_05-08-26.xlsx  # Lista IBrX-100 com pesos
│   └── processed/
│       ├── fatores_fundamentalistas.csv  # P/L, P/VP, ROE, etc.
│       ├── textos_earnings.csv           # Índice dos textos extraídos
│       ├── textos_earnings/              # .txt por empresa/trimestre
│       ├── scores_sentimento.csv         # Scores LLM + citações
│       ├── auditoria_citacoes.csv        # Resultado do citation tracking
│       ├── dataset_final.csv             # Dataset completo para regressão
│       ├── betas_regressao.csv           # Betas OLS+HAC
│       ├── retornos_esperados.csv        # Retorno esperado por empresa/período
│       └── pesos_bl.csv                  # Pesos Black-Litterman por período
│
├── pdfs_coletados/               # PDFs de earnings calls (não versionado)
│   └── TICKER/ANO/TRIMESTRE/
│
├── src/
│   ├── coleta/
│   │   ├── coletar_fundamentais.py   # CVM + yfinance → múltiplos trimestrais
│   │   ├── extrair_texto.py          # (também em sentimento/)
│   │   └── criar_estrutura_pdfs.py   # Cria pastas TICKER/ANO/TRI/
│   │
│   ├── sentimento/
│   │   ├── extrair_texto.py          # pdfplumber + pypdf → .txt
│   │   ├── score_sentimento.py       # LLM → score [-1,+1] + citações
│   │   └── validar_citacoes.py       # Citation tracking determinístico
│   │
│   ├── processamento/
│   │   └── merge_datasets.py         # Junta fundamentais + sentimento
│   │
│   ├── fatores/
│   │   ├── factor_builder.py         # Regressão OLS+HAC + retornos esperados
│   │   ├── normalizacao.py           # Z-score cross-sectional
│   │   └── winsorization.py          # Winsorização por trimestre
│   │
│   ├── modelos/
│   │   └── black_litterman.py        # BL: prior + views → pesos ótimos
│   │
│   └── validacao/
│       └── backtest.py               # Backtest OOS: Spearman + alpha
│
└── outputs/
    ├── ranking_bl.csv                # Ranking BL por período
    ├── tabela_fatores.csv            # Tabela completa para relatório
    ├── backtest_resultados.csv       # Resultados por período
    ├── backtest_resumo.txt           # Resumo do backtest
    └── grounding_resumo.txt          # Resumo do citation tracking
```

---

## Dados

### Empresas cobertas (15 do IBrX-100)

| Ticker | Empresa | Peso IBrX-100 |
|---|---|---|
| VALE3 | Vale | 12.1% |
| ITUB4 | Itaú Unibanco | 8.4% |
| PETR4 | Petrobras | 7.0% |
| AXIA3 | Axia Energia | 4.3% |
| SBSP3 | Sabesp | 3.6% |
| BBDC4 | Bradesco | 3.6% |
| ITSA4 | Itaúsa | 3.0% |
| B3SA3 | B3 | 2.9% |
| WEGE3 | WEG | 2.7% |
| ABEV3 | Ambev | 2.5% |
| BPAC11 | BTG Pactual | 2.5% |
| EMBJ3 | Embraer | 2.4% |
| BBAS3 | Banco do Brasil | 2.2% |
| ENEV3 | Eneva | 1.9% |
| RENT3 | Localiza | 1.4% |

**Cobertura total:** ~52.5% do IBrX-100 por peso

### Período
- **Fundamentais (CVM):** 1T23 a 3T25 (9 trimestres, sem 4T — ITR não inclui)
- **Sentimento (PDFs):** 1T23 a 4T25 (181 PDFs coletados manualmente)

### Fontes
| Dado | Fonte |
|---|---|
| Preços históricos | yfinance (.SA) |
| Demonstrativos financeiros | CVM dados abertos (ITR) |
| Pesos do índice | B3 (IBXXDia_05-08-26.xlsx) |
| Earnings calls / releases | RI das empresas (coleta manual) |
| Sentimento | Gemini 2.0-flash / Claude Sonnet |

---

## Metodologia

### 1. Coleta de Fundamentais

Lê os arquivos ITR da CVM (`itr_cia_aberta_DRE/BPA/BPP_con_ANO.csv`) e cruza com preços históricos do yfinance para calcular os múltiplos trimestrais. Valores da CVM estão em milhares de reais e são convertidos para reais unitários.

**Correções implementadas:**
- `ORDEM_EXERC`: filtro removido (encoding inconsistente causava perda de dados). Usa `drop_duplicates(keep="last")` por empresa/data/conta
- Valores em MIL → multiplicados por 1.000
- EMBR3 → EMBJ3 (migração para Novo Mercado em 2023)
- Download de preços individual por ticker (não em lote — falha silenciosa para alguns .SA)

### 2. Extração de Sentimento

Para cada PDF coletado:
1. **Extração de texto:** `pdfplumber` (primário) + `pypdf` (fallback)
2. **LLM:** classifica tom em 5 categorias (muito_positivo / positivo / neutro / cauteloso / negativo) e extrai até 3 citações literais como evidência
3. **Score numérico:** muito_positivo=+1.0, positivo=+0.5, neutro=0.0, cauteloso=-0.5, negativo=-1.0
4. **Normalização cross-sectional:** rank percentil dentro de cada trimestre — resolve o viés de positividade estrutural em comunicações corporativas

### 3. Citation Tracking

Verificação determinística de cada citação do LLM contra o texto-fonte:

**3 correções implementadas (herdadas do Case 1 do BBI):**
1. **Janela sem teto fixo:** o bug original limitava a busca a 600 chars, rejeitando citações longas legítimas
2. **Normalização robusta:** aspas curvas vs. retas, inserções editoriais entre colchetes, pontuação de fechamento
3. **Segmentação por reticências:** citações compostas verificadas segmento por segmento

**Taxa de grounding final: 84.9%** (498 citações, 423 confirmadas)

### 4. Regressão OLS+HAC

- **Erros HAC (Newey-West, 2 lags):** corrige heterocedasticidade e autocorrelação em séries trimestrais
- **Pooled:** todas as empresas e períodos juntos
- **Z-score cross-sectional:** aplicado antes da regressão para comparabilidade entre fatores de escalas diferentes

### 5. Black-Litterman

- **Prior:** retornos de equilíbrio implícitos do IBrX-100 via CAPM reverso (δ=2.5, pesos reais do índice)
- **Views:** rank percentil do retorno esperado da regressão, convertido para escala do prior (prior_médio ± spread)
- **Parâmetros:** τ=1.0, ω=0.01 (alta confiança nas views)
- **Otimização MVO:** long-only, peso máximo de 20% por ação

### 6. Backtest Out-of-Sample

- **Janela expansível:** betas estimados com dados até T, aplicados em T+1
- **Nunca vê o futuro:** o modelo não usa informação que não estaria disponível no momento da decisão
- **Benchmarks:** IBrX-100 equal weight, portfólio BL
- **Portfólio adicional:** Long Top 5 (5 melhores empresas do ranking com peso igual)

---

## Limitações

**1. N=15 empresas (52.5% do IBrX-100)**
A limitação mais fundamental. Com 15 empresas e 9 trimestres, o poder estatístico é insuficiente para detectar efeitos de valuation pequenos. Fatores de value e quality não são individualmente significativos — resultado consistente com ambiente de juro real alto que comprime múltiplos de forma uniforme (2023-2025).

**2. Ausência do 4T**
O ITR (Informações Trimestrais) não inclui o 4T — esse dado só está no DFP (anual). O 4T é o trimestre mais importante do ano (resultado anual, guidance). O pipeline está estruturado para incluir DFP quando disponível.

**3. Views do Black-Litterman em escala real (resolvido na v2)**
Na versão inicial, as views eram construídas como rank percentil convertido para escala do prior — uma aproximação que gerava alpha negativo. Na versão atual, as views são construídas a partir de uma regressão auxiliar sem z-score, gerando retornos esperados em escala real. O alpha BL passou de -0.09% para +1.78%/trimestre.

**4. Grounding parcial em EMBJ3 (33%) e WEGE3 (28%)**
PDFs dessas empresas têm extração de texto com ruído (tabelas, formatação especial), reduzindo a qualidade das citações do LLM nesses tickers especificamente.

**5. Shares outstanding atuais**
O yfinance retorna o número de ações atuais, não o histórico. Introduces viés pequeno nos múltiplos históricos para empresas que fizeram follow-on ou recompra no período.

---

## Próximos Passos

1. **Expandir para IBrX-100 completo** — pipeline está estruturado, precisa apenas de coleta de PDFs e download de fundamentais para as 86 empresas restantes
2. **Incluir 4T** via DFP anual — aumenta de 9 para 12 trimestres por empresa
3. **Calibrar τ e Ω do BL** via backtest em vez de parâmetros fixos
4. **Testar estabilidade dos betas** com janela rolante
5. **Comparação setorial** — rodar modelo separado por setor (financeiro, utilities, consumo)
6. **Submeter como paper** — SBFin 2027 ou similar (gap real na literatura de NLP financeiro brasileiro)

---

## Referências

- Lee, Y. et al. (2025). *LLM-Enhanced Black-Litterman Portfolio Optimization.* CIKM'25 Workshop on FinAI. arXiv:2504.14345
- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization.* Financial Analysts Journal, 48(5), 28–43
- Fama, E. & French, K. (2015). *A Five-Factor Asset Pricing Model.* Journal of Financial Economics, 116(1), 1–22
- Newey, W. & West, K. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica, 55(3), 703–708
- Gu, S., Kelly, B. & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies, 33(5), 2223–2273
- He, G. & Litterman, R. (1999). *The Intuition Behind Black-Litterman Model Portfolios.* Goldman Sachs Investment Management

---

## Identidade do Robô

**Nome:** CARCARÁ
**Tagline:** *Precisão de rapina. Inteligência de mercado.*

O Carcará é uma ave de rapina do Nordeste brasileiro — veloz, preciso e oportunista. Identifica sua presa antes dos concorrentes e age com convicção. O modelo compartilha esse DNA: varre o IBrX-100 em busca de ações sistematicamente subprecificadas, com disciplina estatística e visão de longo prazo.

