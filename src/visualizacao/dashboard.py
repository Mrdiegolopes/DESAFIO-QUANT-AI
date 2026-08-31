"""
Fortal Gradient — Dashboard Streamlit
Visualização interativa dos resultados do modelo.

Uso: streamlit run src/visualizacao/dashboard.py
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))


import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CARCARÁ — Quant AI Dashboard",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

DIR_PROC = BASE_DIR / "data" / "processed"
DIR_OUT  = BASE_DIR / "outputs"

# ─────────────────────────────────────────────
# CARREGA DADOS
# ─────────────────────────────────────────────

@st.cache_data
def carregar_dados():
    dados = {}
    arquivos = {
        "dataset":    DIR_PROC / "dataset_final.csv",
        "betas":      DIR_PROC / "betas_regressao.csv",
        "retornos":   DIR_PROC / "retornos_esperados.csv",
        "pesos_bl":   DIR_PROC / "pesos_bl.csv",
        "ranking_bl": DIR_OUT  / "ranking_bl.csv",
        "backtest":   DIR_OUT  / "backtest_resultados.csv",
        "auditoria":  DIR_PROC / "auditoria_citacoes.csv",
        "sentimento": DIR_PROC / "scores_sentimento.csv",
    }
    for nome, path in arquivos.items():
        if path.exists():
            dados[nome] = pd.read_csv(path, encoding="utf-8-sig")
        else:
            dados[nome] = None
    return dados

dados = carregar_dados()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Caracara_plancus_-_Southern_Caracara.jpg/320px-Caracara_plancus_-_Southern_Caracara.jpg",
                 caption="Fortal Gradient", width="stretch")
st.sidebar.title("🦅 Fortal Gradient")
st.sidebar.caption("*Precisão de rapina. Inteligência de mercado.*")
st.sidebar.divider()

aba = st.sidebar.radio(
    "Navegação",
    ["📊 Visão Geral", "📈 Backtest", "🧮 Fatores", "💼 Portfólio BL",
     "🤖 Sentimento", "✅ Citation Tracking"],
)

# ─────────────────────────────────────────────
# ABA: VISÃO GERAL
# ─────────────────────────────────────────────

if aba == "📊 Visão Geral":
    st.title("🦅 CARCARÁ — Quality-Adjusted Value + LLM Sentiment")
    st.caption("Desafio Quant AI Itaú Asset 2026 | IBrX-100")
    st.divider()

    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Spearman OOS", "0.183", "full-sample: 0.264")
    with col2:
        st.metric("Alpha BL vs EW", "+1.78%/tri", "+7.1%/ano")
    with col3:
        st.metric("Sentimento p-valor", "0.033", " Significativo (**)")
    with col4:
        st.metric("Citation Tracking", "84.9%", "498 citações verificadas")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Arquitetura do Pipeline")
        st.code("""
CVM (ITR) + yfinance
    → Fundamentais trimestrais
    
PDFs earnings calls (181)
    → Extração de texto
    → LLM sentimento [-1, +1]
    → Citation tracking (84.9%)
    
OLS+HAC (Newey-West)
    → β sentimento p=0.033 **
    → R²=6.4%
    
Black-Litterman
    → Views rank percentil
    → Pesos ótimos MVO
    
Backtest OOS (janela expansível)
    → Spearman=0.183
    → Alpha Top 5=+1.71%/tri
        """, language="text")

    with col_b:
        st.subheader("Empresas Cobertas (15)")
        if dados["dataset"] is not None:
            tickers = dados["dataset"]["ticker"].unique()
            pesos = {
                "VALE3":0.121,"ITUB4":0.084,"PETR4":0.070,
                "AXIA3":0.043,"SBSP3":0.037,"BBDC4":0.036,
                "ITSA4":0.030,"B3SA3":0.029,"WEGE3":0.027,
                "ABEV3":0.025,"BPAC11":0.025,"EMBJ3":0.024,
                "BBAS3":0.022,"ENEV3":0.019,"RENT3":0.014,
            }
            df_emp = pd.DataFrame([
                {"Ticker": t, "Peso IBrX-100": pesos.get(t, 0)}
                for t in sorted(tickers)
            ]).sort_values("Peso IBrX-100", ascending=False)
            fig = px.bar(df_emp, x="Ticker", y="Peso IBrX-100",
                        color="Peso IBrX-100", color_continuous_scale="Blues")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────
# ABA: BACKTEST
# ─────────────────────────────────────────────

elif aba == "📈 Backtest":
    st.title("📈 Backtest Out-of-Sample")
    st.caption("Retorno acumulado e métricas OOS por período")
    st.divider()

    if dados["backtest"] is None:
        st.warning("backtest_resultados.csv não encontrado. Rode src/validacao/backtest.py primeiro.")
    else:
        df_bt = dados["backtest"]

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spearman Full-Sample", "0.264", "inflacionado")
        with col2:
            st.metric("Spearman OOS Médio", f"{df_bt['spearman_oos'].mean():.3f}")
        with col3:
            st.metric("Alpha Top 5 vs EW", f"+{df_bt['alpha_top5_vs_ew'].mean()*100:.2f}%/tri")
        with col4:
            ganhou = (df_bt["alpha_top5_vs_ew"] > 0).sum()
            st.metric("Top 5 bateu EW", f"{ganhou}/{len(df_bt)} períodos")

        st.divider()

        # Gráfico de retornos acumulados
        df_bt_sorted = df_bt.sort_values("periodo")
        df_bt_sorted["cum_top5"] = (1 + df_bt_sorted["retorno_top5"]).cumprod()
        df_bt_sorted["cum_ew"]   = (1 + df_bt_sorted["retorno_ew"]).cumprod()
        df_bt_sorted["cum_bl"]   = (1 + df_bt_sorted["retorno_bl"]).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_bt_sorted["periodo"], y=df_bt_sorted["cum_top5"],
                                  name="Top 5", line=dict(color="#FF6B35", width=3)))
        fig.add_trace(go.Scatter(x=df_bt_sorted["periodo"], y=df_bt_sorted["cum_bl"],
                                  name="Black-Litterman", line=dict(color="#2E86AB", width=2)))
        fig.add_trace(go.Scatter(x=df_bt_sorted["periodo"], y=df_bt_sorted["cum_ew"],
                                  name="Equal Weight", line=dict(color="#A0A0A0", width=2, dash="dash")))
        fig.update_layout(title="Retorno Acumulado OOS", height=400,
                         yaxis_title="Retorno Acumulado", xaxis_title="Período")
        st.plotly_chart(fig, width="stretch")

        # Tabela por período
        st.subheader("Resultados por Período")
        cols_show = ["periodo","spearman_oos","retorno_top5","retorno_bl",
                     "retorno_ew","alpha_top5_vs_ew"]
        df_show = df_bt[cols_show].copy()
        df_show.columns = ["Período","Spearman OOS","Top 5","BL","EW","Alpha Top5"]
        for col in ["Top 5","BL","EW","Alpha Top5"]:
            df_show[col] = df_show[col].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
        st.dataframe(df_show, width="stretch")

        # Spearman por período
        fig2 = px.bar(df_bt_sorted, x="periodo", y="spearman_oos",
                      color="spearman_oos", color_continuous_scale="RdYlGn",
                      title="Spearman OOS por Período")
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, width="stretch")

# ─────────────────────────────────────────────
# ABA: FATORES
# ─────────────────────────────────────────────

elif aba == "🧮 Fatores":
    st.title("🧮 Motor de Fatores — OLS+HAC")
    st.caption("Regressão pooled com erros Newey-West (2 lags)")
    st.divider()

    if dados["betas"] is None:
        st.warning("betas_regressao.csv não encontrado.")
    else:
        df_b = dados["betas"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²", f"{df_b['r2'].iloc[0]:.1%}")
        with col2:
            st.metric("N observações", int(df_b["n_obs"].iloc[0]))
        with col3:
            st.metric("Durbin-Watson", f"{df_b['dw'].iloc[0]:.3f}")

        st.divider()

        # Tabela de betas
        df_b_show = df_b[["nome","beta","tstat","pvalue"]].copy()
        df_b_show.columns = ["Fator","Beta","t-stat","p-valor"]
        df_b_show["Significância"] = df_b_show["p-valor"].apply(
            lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "")
        st.dataframe(df_b_show, width="stretch")

        # Gráfico de betas
        fig = px.bar(df_b, x="nome", y="beta",
                     color="pvalue",
                     color_continuous_scale="RdYlGn_r",
                     title="Betas da Regressão OLS+HAC",
                     labels={"nome":"Fator","beta":"Beta","pvalue":"p-valor"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

        # Retornos esperados por empresa
        if dados["retornos"] is not None:
            st.subheader("Retorno Esperado por Empresa — Último Período")
            ultimo = dados["retornos"]["periodo"].max()
            df_ult = dados["retornos"][dados["retornos"]["periodo"] == ultimo].copy()
            df_ult = df_ult.sort_values("retorno_esperado", ascending=False)
            fig2 = px.bar(df_ult, x="ticker", y="retorno_esperado",
                          color="retorno_esperado", color_continuous_scale="RdYlGn",
                          title=f"Ranking de Retorno Esperado — {ultimo}")
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, width="stretch")

# ─────────────────────────────────────────────
# ABA: PORTFÓLIO BL
# ─────────────────────────────────────────────

elif aba == "💼 Portfólio BL":
    st.title("💼 Portfólio Black-Litterman")
    st.divider()

    if dados["ranking_bl"] is None:
        st.warning("ranking_bl.csv não encontrado.")
    else:
        df_rank = dados["ranking_bl"]
        periodos = sorted(df_rank["periodo"].unique())
        periodo_sel = st.selectbox("Selecione o período", periodos, index=len(periodos)-1)

        df_per = df_rank[df_rank["periodo"] == periodo_sel].copy()
        df_per = df_per.sort_values("mu_bl", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Retorno Esperado BL")
            fig = px.bar(df_per, x="ticker", y="mu_bl",
                         color="mu_bl", color_continuous_scale="RdYlGn",
                         title=f"μ_BL — {periodo_sel}")
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Pesos do Portfólio")
            df_peso = df_per[df_per["peso_bl"] > 0.001]
            fig2 = px.pie(df_peso, values="peso_bl", names="ticker",
                          title=f"Alocação — {periodo_sel}")
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, width="stretch")

        st.subheader("Ranking Completo")
        st.dataframe(df_per[["rank","ticker","mu_bl","peso_bl"]].style.format(
            {"mu_bl": "{:.4f}", "peso_bl": "{:.2%}"}), width="stretch")

# ─────────────────────────────────────────────
# ABA: SENTIMENTO
# ─────────────────────────────────────────────

elif aba == "🤖 Sentimento":
    st.title("🤖 Análise de Sentimento LLM")
    st.caption("Score normalizado cross-sectionally por trimestre")
    st.divider()

    if dados["sentimento"] is None:
        st.warning("scores_sentimento.csv não encontrado.")
    else:
        df_s = dados["sentimento"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documentos analisados", len(df_s))
        with col2:
            positivos = (df_s["tom"].isin(["muito_positivo","positivo"])).sum()
            st.metric("Tom positivo", f"{positivos/len(df_s):.0%}")
        with col3:
            st.metric("Score médio", f"{df_s['score'].mean():.2f}")

        st.divider()

        # Distribuição de tons
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(df_s, x="tom", color="tom",
                               title="Distribuição de Tons",
                               category_orders={"tom": ["muito_positivo","positivo",
                                                          "neutro","cauteloso","negativo"]})
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, width="stretch")

        with col_b:
            score_ticker = df_s.groupby("ticker")["score"].mean().reset_index()
            score_ticker = score_ticker.sort_values("score", ascending=False)
            fig2 = px.bar(score_ticker, x="ticker", y="score",
                          color="score", color_continuous_scale="RdYlGn",
                          title="Score Médio por Empresa")
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, width="stretch")

        # Heatmap de sentimento por empresa/período
        if "periodo" in df_s.columns:
            pivot = df_s.pivot_table(
                values="score", index="ticker", columns="periodo", aggfunc="mean")
            fig3 = px.imshow(pivot, color_continuous_scale="RdYlGn",
                             title="Heatmap de Sentimento — Ticker × Período",
                             aspect="auto")
            fig3.update_layout(height=450)
            st.plotly_chart(fig3, width="stretch")

# ─────────────────────────────────────────────
# ABA: CITATION TRACKING
# ─────────────────────────────────────────────

elif aba == "✅ Citation Tracking":
    st.title("✅ Citation Tracking")
    st.caption("Verificação determinística das citações do LLM contra o texto-fonte")
    st.divider()

    if dados["auditoria"] is None:
        st.warning("auditoria_citacoes.csv não encontrado.")
    else:
        df_a = dados["auditoria"]
        total = len(df_a)
        enc   = (df_a["status"] == "encontrada").sum()
        apr   = (df_a["status"] == "aproximada").sum()
        nao   = (df_a["status"] == "nao_encontrada").sum()
        grounding = (enc + apr) / total

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Taxa de Grounding", f"{grounding:.1%}")
        with col2:
            st.metric("Encontradas", f"{enc} ({enc/total:.0%})")
        with col3:
            st.metric("Aproximadas", f"{apr} ({apr/total:.0%})")
        with col4:
            st.metric("Não encontradas", f"{nao} ({nao/total:.0%})")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            # Pizza de status
            fig = px.pie(
                values=[enc, apr, nao],
                names=["Encontrada","Aproximada","Não encontrada"],
                color_discrete_sequence=["#2ECC71","#F39C12","#E74C3C"],
                title="Status das Citações",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")

        with col_b:
            # Grounding por ticker
            grounding_ticker = df_a.groupby("ticker").apply(
                lambda g: (g["status"].isin(["encontrada","aproximada"])).mean()
            ).reset_index()
            grounding_ticker.columns = ["ticker","grounding"]
            grounding_ticker = grounding_ticker.sort_values("grounding", ascending=False)
            fig2 = px.bar(grounding_ticker, x="ticker", y="grounding",
                          color="grounding", color_continuous_scale="RdYlGn",
                          title="Taxa de Grounding por Empresa")
            fig2.add_hline(y=grounding, line_dash="dash",
                           annotation_text=f"Média: {grounding:.1%}")
            fig2.update_layout(height=350, yaxis_tickformat=".0%")
            st.plotly_chart(fig2, width="stretch")

        st.subheader("Citações Não Encontradas")
        df_nao = df_a[df_a["status"] == "nao_encontrada"][
            ["ticker","periodo","coluna","citacao","similaridade"]]
        st.dataframe(df_nao, width="stretch")