import os
import zipfile
import requests
from pathlib import Path
import pandas as pd

# Configuração de caminhos
BASE_DIR = Path(__file__).resolve().parents[2]
PASTA_CVM_RAW = BASE_DIR / "data" / "raw" / "cvm"
PASTA_CVM_RAW.mkdir(parents=True, exist_ok=True)

# URL base do portal de dados abertos da CVM para ITR (Informações Trimestrais)
URL_BASE_ITR = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/"

# Tipos de demonstrativos que nos interessam (Consolidados)
DEMONSTRATIVOS = ["BPA_con", "BPP_con", "DRE_con"]


def baixar_itr_cvm(anos: list[int]):
    """
    Baixa e extrai os demonstrativos ITR da CVM para a lista de anos fornecida.
    """
    print("=" * 60)
    print("INICIANDO COLETA DE DADOS FINANCEIROS CVM (ITR)")
    print("=" * 60)

    for ano in anos:
        nome_zip = f"itr_cia_aberta_{ano}.zip"
        url = f"{URL_BASE_ITR}{nome_zip}"
        caminho_zip = PASTA_CVM_RAW / nome_zip

        print(f"\n[+] Baixando arquivo ITR do ano {ano}...")
        try:
            resposta = requests.get(url, stream=True, timeout=30)
            if resposta.status_code == 200:
                with open(caminho_zip, "wb") as f:
                    for chunk in resposta.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"    [ok] Download concluído: {nome_zip}")

                # Extraindo apenas os demonstrativos consolidados
                with zipfile.ZipFile(caminho_zip, "r") as zip_ref:
                    arquivos_no_zip = zip_ref.namelist()
                    for arquivo in arquivos_no_zip:
                        if any(demo in arquivo for demo in DEMONSTRATIVOS):
                            zip_ref.extract(arquivo, PASTA_CVM_RAW)
                            print(f"    [ok] Extraído: {arquivo}")

                # Remove o arquivo .zip para economizar espaço
                caminho_zip.unlink()

            else:
                print(f"    [x] Falha ao baixar {ano} - Status Code: {resposta.status_code}")

        except Exception as e:
            print(f"    [x] Erro ao processar ano {ano}: {e}")


def consolidar_demonstrativos(anos: list[int]):
    print("\n" + "=" * 60)
    print("CONSOLIDANDO DEMONSTRATIVOS HISTÓRICOS")


    for demo in DEMONSTRATIVOS:
        dfs = []
        for ano in anos:
            nome_arquivo = f"itr_cia_aberta_{demo}_{ano}.csv"
            caminho_arquivo = PASTA_CVM_RAW / nome_arquivo

            if caminho_arquivo.exists():
                try:
                    df = pd.read_csv(
                        caminho_arquivo,
                        sep=";",
                        encoding="ISO-8859-1",
                        dtype={"CD_CVM": str, "CNPJ_CIA": str}
                    )
                    dfs.append(df)
                except Exception as e:
                    print(f"    [x] Erro ao ler {nome_arquivo}: {e}")

        if dfs:
            df_final = pd.concat(dfs, ignore_index=True)
            arquivo_saida = PASTA_CVM_RAW / f"cvm_historico_{demo.lower()}.csv"
            df_final.to_csv(arquivo_saida, index=False, sep=";", encoding="utf-8")
            print(f"    [ok] Consolidado salvo em: {arquivo_saida.name} (Linhas: {len(df_final):,})")

if __name__ == "__main__":
    # Coletar dados de 2019 até 2026
    ANOS_COLETA = list(range(2019, 2027))
    
    baixar_itr_cvm(ANOS_COLETA)
    consolidar_demonstrativos(ANOS_COLETA)