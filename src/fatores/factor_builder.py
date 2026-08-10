import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao sys.path do Python
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import pandas as pd
from src.fatores.winsorization import winsorize_factors_by_quarter
from src.fatores.normalizacao import zscore_cross_sectional

PASTA_PROCESSED = BASE_DIR / "data" / "processed"
FATORES_VALUATION = ["P_L", "P_VP", "EV_EBIT"]


def construir_fatores():
    print("=" * 60)
    print("CONSTRUINDO E SANITIZANDO FATORES QUANTITATIVOS (Z-SCORES)")
    print("=" * 60)

    arq_multiplos = PASTA_PROCESSED / "multiplos_valuation_trimestrais.csv"
    if not arq_multiplos.exists():
        raise FileNotFoundError(f"Arquivo {arq_multiplos} não encontrado. Execute calcular_multiplos.py primeiro.")

    df = pd.read_csv(arq_multiplos)

    print(f"[+] Registros carregados: {len(df):,}")

    # 1. Winsorização por trimestre (Caudas 1% e 99%)
    print("[+] Aplicando Winsorização (1% / 99%) por trimestre...")
    df_winsor = winsorize_factors_by_quarter(df, FATORES_VALUATION, date_col="DT_REFER")

    # 2. Padronização Cross-Sectional (Z-Score)
    print("[+] Calculando Z-Scores Cross-Sectional...")
    df_fatores = zscore_cross_sectional(df_winsor, FATORES_VALUATION, date_col="DT_REFER")

    saida = PASTA_PROCESSED / "fatores_valuation_sanitizados.csv"
    df_fatores.to_csv(saida, index=False)
    print(f"\n[✔] Tabela de fatores sanitizados salva em: {saida} ({len(df_fatores):,} registros)")


if __name__ == "__main__":
    construir_fatores()