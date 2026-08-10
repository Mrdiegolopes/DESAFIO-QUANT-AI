from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PASTA_CVM_RAW = BASE_DIR / "data" / "raw" / "cvm"
PASTA_PROCESSED = BASE_DIR / "data" / "processed"
PASTA_PROCESSED.mkdir(parents=True, exist_ok=True)

CONTAS_ALVO = {
    "2.03": "patrimonio_liquido",
    "3.11": "lucro_liquido",
    "3.05": "ebit",
    "1.01.01": "caixa_equivalentes",
    "2.01.04": "divida_curto_prazo",
    "2.02.01": "divida_longo_prazo"
}


def processar_arquivo_cvm(caminho_csv: Path) -> pd.DataFrame:
    """Lê arquivo CVM consolidado com auto-detecção de separador e extrai contas-chave."""
    # sep=None com engine="python" identifica se o arquivo usa ',' ou ';'
    df = pd.read_csv(
        caminho_csv,
        sep=None,
        engine="python",
        encoding="utf-8",
        dtype=str
    )

    # Caso o arquivo antigo estivesse em ISO-8859-1
    if len(df.columns) == 1:
        df = pd.read_csv(
            caminho_csv,
            sep=None,
            engine="python",
            encoding="ISO-8859-1",
            dtype=str
        )

    # Padronizar nomes das colunas
    df.columns = df.columns.str.strip().str.upper()

    if "CD_CVM" not in df.columns:
        raise KeyError(f"Coluna 'CD_CVM' não encontrada. Colunas lidas: {list(df.columns)}")

    # Garantir formatação do código CVM
    df["CD_CVM"] = df["CD_CVM"].str.strip().str.zfill(6)
    df["CD_CONTA"] = df["CD_CONTA"].str.strip()

    # Converter valor da conta para numérico
    df["VL_CONTA"] = pd.to_numeric(df["VL_CONTA"].str.replace(",", "."), errors="coerce")

    # 1. Filtrar exercício recente
    if "ORDEM_EXERC" in df.columns:
        df = df[df["ORDEM_EXERC"] == "ÚLTIMO"]

    # 2. Ordenar e manter a versão mais recente
    colunas_ordem = [c for c in ["CD_CVM", "DT_REFER", "VERSAO"] if c in df.columns]
    df = df.sort_values(by=colunas_ordem)

    # 3. Filtrar contas alvo
    df = df[df["CD_CONTA"].isin(CONTAS_ALVO.keys())].copy()
    df["rubrica"] = df["CD_CONTA"].map(CONTAS_ALVO)

    # Pegar o último registro por grupo
    df = df.groupby(
        ["CD_CVM", "DENOM_CIA", "CNPJ_CIA", "DT_REFER", "rubrica"],
        as_index=False
    ).last()

    # 4. Pivotar para formato wide
    df_pivot = df.pivot(
        index=["CD_CVM", "DENOM_CIA", "CNPJ_CIA", "DT_REFER"],
        columns="rubrica",
        values="VL_CONTA"
    ).reset_index()

    return df_pivot


def consolidar_demonstrativos_processados():
    print("=" * 60)
    print("PROCESSANDO DEMONSTRATIVOS FINANCEIROS CVM")
    print("=" * 60)

    dfs = []
    arquivos = list(PASTA_CVM_RAW.glob("cvm_historico_*.csv"))

    if not arquivos:
        print("[x] Nenhum arquivo cvm_historico_*.csv encontrado em data/raw/cvm/")
        return

    for arq in arquivos:
        print(f"[+] Lendo e filtrando: {arq.name}")
        try:
            df_proc = processar_arquivo_cvm(arq)
            dfs.append(df_proc)
        except Exception as e:
            print(f"    [x] Erro ao processar {arq.name}: {e}")

    if dfs:
        df_consolidado = pd.concat(dfs, ignore_index=True)

        df_final = df_consolidado.groupby(
            ["CD_CVM", "DENOM_CIA", "CNPJ_CIA", "DT_REFER"], as_index=False
        ).first()

        colunas_rubricas = list(CONTAS_ALVO.values())
        for col in colunas_rubricas:
            if col not in df_final.columns:
                df_final[col] = 0.0
            else:
                df_final[col] = df_final[col].fillna(0.0)

        df_final["divida_liquida"] = (
            df_final["divida_curto_prazo"] + df_final["divida_longo_prazo"]
        ) - df_final["caixa_equivalentes"]

        saida = PASTA_PROCESSED / "cvm_demonstrativos_limpos.csv"
        df_final.to_csv(saida, index=False, encoding="utf-8")
        print(f"\n[ok] Tabela limpa salva em: {saida} ({len(df_final):,} registros)")


if __name__ == "__main__":
    consolidar_demonstrativos_processados()