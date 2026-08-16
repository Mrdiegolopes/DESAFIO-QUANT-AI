"""
Fortal Gradient — Pipeline Principal
Quality-Adjusted Value Factor Model with LLM Sentiment + Black-Litterman

Executa o pipeline completo em sequência ou módulos individuais.

Uso:
    python main.py                    # pipeline completo
    python main.py --etapa coleta     # só coleta de fundamentais
    python main.py --etapa sentimento # só extração de sentimento
    python main.py --etapa fatores    # só motor de fatores
    python main.py --etapa bl         # só Black-Litterman
    python main.py --etapa backtest   # só backtest
    python main.py --dashboard        # abre interface Streamlit
"""

import sys
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ETAPAS = {
    "coleta":     "src.coleta.coletar_fundamentais",
    "texto":      "src.sentimento.extrair_texto",
    "sentimento": "src.sentimento.score_sentimento",
    "citacoes":   "src.sentimento.validar_citacoes",
    "merge":      "src.processamento.merge_datasets",
    "fatores":    "src.fatores.factor_builder",
    "bl":         "src.modelos.black_litterman",
    "backtest":   "src.validacao.backtest",
}


def rodar_etapa(nome: str):
    """Importa e executa o main() de um módulo."""
    modulo_path = ETAPAS.get(nome)
    if not modulo_path:
        log.error(f"Etapa desconhecida: {nome}. Opções: {list(ETAPAS.keys())}")
        return False

    log.info(f"\n{'='*60}")
    log.info(f"ETAPA: {nome.upper()}")
    log.info(f"{'='*60}")

    try:
        import importlib
        modulo = importlib.import_module(modulo_path)
        if hasattr(modulo, "main"):
            modulo.main()
        else:
            log.warning(f"Módulo {modulo_path} não tem função main()")
        return True
    except Exception as e:
        log.error(f"Erro na etapa {nome}: {e}")
        import traceback
        traceback.print_exc()
        return False


def pipeline_completo():
    """Executa todas as etapas em sequência."""
    log.info("🦅 CARCARÁ — Pipeline Completo")
    log.info("Quality-Adjusted Value Factor + LLM Sentiment + Black-Litterman")

    ordem = [
        "coleta",
        "texto",
        "sentimento",
        "citacoes",
        "merge",
        "fatores",
        "bl",
        "backtest",
    ]

    for etapa in ordem:
        sucesso = rodar_etapa(etapa)
        if not sucesso:
            log.error(f"Pipeline interrompido na etapa: {etapa}")
            log.error("Corrija o erro e rode novamente com --etapa {etapa}")
            sys.exit(1)

    log.info("\n" + "="*60)
    log.info("✅ PIPELINE COMPLETO")
    log.info("="*60)
    log.info("Resultados salvos em outputs/")
    log.info("Para visualizar: python main.py --dashboard")


def abrir_dashboard():
    """Abre a interface Streamlit."""
    import subprocess
    dashboard = BASE_DIR / "src" / "visualizacao" / "dashboard.py"
    if not dashboard.exists():
        log.error(f"Dashboard não encontrado: {dashboard}")
        log.error("Crie src/visualizacao/dashboard.py primeiro")
        sys.exit(1)
    log.info("Abrindo dashboard Streamlit...")
    subprocess.run(["streamlit", "run", str(dashboard)], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="CARCARÁ — Pipeline Quant AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--etapa",
        choices=list(ETAPAS.keys()),
        help="Executa uma etapa específica do pipeline",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Abre a interface Streamlit",
    )
    parser.add_argument(
        "--listar",
        action="store_true",
        help="Lista todas as etapas disponíveis",
    )

    args = parser.parse_args()

    if args.listar:
        print("\nEtapas disponíveis:")
        for nome, modulo in ETAPAS.items():
            print(f"  {nome:<12} → {modulo}")
        return

    if args.dashboard:
        abrir_dashboard()
        return

    if args.etapa:
        rodar_etapa(args.etapa)
        return

    # Sem argumentos — roda pipeline completo
    pipeline_completo()


if __name__ == "__main__":
    main()