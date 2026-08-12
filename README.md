## LLM-Enhanced Black-Litterman Portfolio Optimization via Value Factor Model & Verified Earnings Call Sentiment in IBrX-100

Coleta de dados 
Lista de tickers
        │
        ▼
Preços históricos (Yahoo Finance)
        │
        ▼
Dados financeiros (CVM)
        │
        ▼
Cálculo dos múltiplos
        │
        ▼
Setor das empresas
        │
        ▼
Earnings Calls
        │
        ▼
LLM



Esboço da proposta 

Valuation histórico + Sentimento (earnings calls)
        ↓
   Regressão OLS+HAC
        ↓
   Retorno esperado por ação (views)
        ↓
   Black-Litterman (blenda views com prior de mercado)
        ↓
   Pesos otimizados do portfólio IBrX-100

   Value Factor Model + Sentimento + Black-Litterman para o IBrX-100

Modelo que usa valuation histórico trimestral (P/L, EV/EBITDA, P/VP) e sentimento de earnings calls como fatores em uma regressão OLS+HAC para gerar views de retorno por ação do IBrX-100. Essas views — com incerteza quantificada pelos R² e p-valores da regressão, não por repetição de LLM — alimentam um otimizador Black-Litterman que as blenda com o prior de equilíbrio de mercado para produzir pesos ótimos de portfólio. Backtest out-of-sample genuíno compara Spearman full-sample vs. out-of-sample, e performance final é medida por Sharpe ratio e MDD contra benchmarks (IBrX-100 igual pesos, MVO clássico).


## Arquitetura do Modelo

                                [ Dados Públicos CVM (ITR) ]
                                              │
  [ Transcrições Earnings Calls ]             ▼
               │                     Múltiplos de Valuation 
               ▼                  (P/L, EV/EBITDA, P/VP, DY)
  Sentimento Extraído via LLM                 │
  + Citation Grounding Check                   │
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
                    Fator Engine: Regressão OLS + HAC (Newey-West)
                    • Regressão com dummy COVID (1T20/2T20)
                    • Estimativa estatística de Retorno Esperado
                               │
                               ▼
                  Views do Black-Litterman (Q)
                  & Matriz de Incerteza (Omega) via R² / p-valores
                               │
                               ▼
            Otimizador Black-Litterman (Prior de Mercado + Views)
                               │
                               ▼
               Pesos Ótimos do Portfólio IBrX-100
                               │
                               ▼
       Backtest Out-of-Sample (Sharpe, MDD, Rank Correlation Spearman)



Limpeza do Código de Preços:
Corrija o salvamento do CSV no script yfinance.

Pipeline CVM (ITR / DFP):Extrair balanços e DREs históricos via dados abertos da CVM para calcular $P/L$, $EV/EBITDA$, $P/VP$ e $Dividend\ Yield$ no nível trimestral de 2015 a 2026.

Módulo de Extração de Sentimento:Estruturar o pipeline de scraping e extração das transcrições de RI / Earnings Calls das empresas do IBrX-100.Implementar a função de verificação de citações (string matching exato da citação retornada pelo LLM na transcrição).

Construção do Motor Fatorial + Black-Litterman:Rodar as regressões OLS com erros HAC (Newey-West) para geração do vetor de retornos esperados ($Q$) e da matriz de incerteza ($\Omega$).Calibrar a matriz de covariância histórica ($\Sigma$) e o parâmetro de aversão ao risco ($\delta$) para a otimização de Black-Litterman.