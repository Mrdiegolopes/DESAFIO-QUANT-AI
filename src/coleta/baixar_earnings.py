"""
Scraper de Earnings Calls — Top 15 IBrX-100 (v2)
Melhorias vs v1:
- Timeout aumentado de 15s para 30s
- Espera de 6s após carregar (era 3s)
- Scroll progressivo para forçar lazy load
- Aceita links sem .pdf (botões de download via JS)
- Tenta clicar em abas/filtros de ano quando disponíveis
- URLs alternativas por empresa como fallback
"""

import os, re, time, hashlib, logging, urllib3, requests
import unicodedata, pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RAIZ              = Path(__file__).parent.parent.parent
DIRETORIO_PDFS    = RAIZ / "pdfs_coletados"
DIRETORIO_LOGS    = RAIZ / "outputs"
CHROMEDRIVER_PATH = RAIZ / "drivers" / "chromedriver.exe"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# URLs alternativas por ticker (fallback se a principal não tiver PDFs)
URLS_ALTERNATIVAS = {
    "VALE3":  [
        "https://ri.vale.com/pt-br/resultados-e-apresentacoes/central-de-resultados",
        "https://ri.vale.com/resultados-e-apresentacoes",
    ],
    "ITUB4":  [
        "https://www.itau.com.br/relacoes-com-investidores/resultados-e-informacoes/central-de-resultados",
        "https://www.itau.com.br/relacoes-com-investidores/resultados",
    ],
    "PETR4":  [
        "https://ri.petrobras.com.br/pt/resultados/central-de-resultados",
        "https://ri.petrobras.com.br/pt/resultados",
        "https://ri.petrobras.com.br/resultados",
    ],
    "PETR3":  [
        "https://ri.petrobras.com.br/pt/resultados/central-de-resultados",
        "https://ri.petrobras.com.br/pt/resultados",
    ],
    "BBDC4":  [
        "https://ri.bradesco.com.br/pt-br/resultados-e-apresentacoes/central-de-resultados",
        "https://ri.bradesco.com.br/pt-br/informacoes-financeiras/resultados",
    ],
    "BBAS3":  [
        "https://ri.bb.com.br/informacoes-financeiras/central-de-resultados",
        "https://ri.bb.com.br/central-de-resultados",
    ],
    "B3SA3":  [
        "https://ri.b3.com.br/pt-br/informacoes-financeiras/central-de-resultados",
        "https://ri.b3.com.br/pt-br/central-de-resultados",
    ],
    "WEGE3":  [
        "https://ri.weg.net/pt-br/informacoes-financeiras/central-de-resultados",
        "https://ri.weg.net/central-de-resultados",
    ],
    "ABEV3":  [
        "https://ri.ambev.com.br/relatorios-publicacoes/divulgacao-de-resultados",
        "https://ri.ambev.com.br/central-de-resultados",
    ],
    "BPAC11": [
        "https://ri.btgpactual.com/principais-informacoes/central-de-resultados",
        "https://ri.btgpactual.com/central-de-resultados",
    ],
    "EMBJ3":  [
        "https://ri.embraer.com.br/informacoes-financeiras/central-de-resultados",
        "https://ri.embraer.com.br/central-de-resultados",
    ],
    "SBSP3":  [
        "https://ri.sabesp.com.br/informacoes-financeiras/central-de-resultados",
        "https://ri.sabesp.com.br/central-de-resultados",
    ],
    "ITSA4":  [
        "https://ri.itausa.com.br/informacoes-financeiras/central-de-resultados",
        "https://ri.itausa.com.br/central-de-resultados",
    ],
    "AXIA3":  [
        "https://ri.enelbrasil.com.br/central-de-resultados",
        "https://ri.axiaenergia.com.br/informacoes-financeiras/central-de-resultados",
    ],
}

KEYWORDS = {
    "transcricao": ["transcri","transcript","teleconfer","videoconfer",
                    "conference call","earnings call","call de resultado"],
    "release":     ["release","press release","resultado","divulga",
                    "earnings","comunicado","nota de resultado"],
    "apresentacao":["apresenta","presentation","slides","webcast","investor"],
}

TRIMESTRE_KEYWORDS = {
    "1T": ["1t","1q","1trim","primeiro trimestre","first quarter",
           "jan","fev","mar","january","february","march"],
    "2T": ["2t","2q","2trim","segundo trimestre","second quarter",
           "abr","mai","jun","april","may","june"],
    "3T": ["3t","3q","3trim","terceiro trimestre","third quarter",
           "jul","ago","set","july","august","september"],
    "4T": ["4t","4q","4trim","quarto trimestre","fourth quarter",
           "out","nov","dez","october","november","december",
           "anual","annual","full year"],
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}


def _norm(t): 
    t = unicodedata.normalize("NFKD", str(t))
    return "".join(c for c in t if not unicodedata.combining(c)).lower().strip()

def _contem(texto, kws): 
    t = _norm(texto)
    return any(_norm(k) in t for k in kws)

def _detectar_tri(texto):
    t = _norm(texto)
    for tri, kws in TRIMESTRE_KEYWORDS.items():
        if any(k in t for k in kws):
            return tri
    return None

def _hash(url): 
    return hashlib.md5(url.encode()).hexdigest()[:8]


def _criar_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--ignore-certificate-errors")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36")
    svc = Service(executable_path=str(CHROMEDRIVER_PATH))
    drv = webdriver.Chrome(service=svc, options=opts)
    drv.execute_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    return drv


def _extrair_links(driver, url_base, ano, trimestre):
    candidatos = {"transcricao": [], "release": [], "apresentacao": []}
    try:
        driver.get(url_base)
        # Espera mais tempo para JS pesado
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "a")))
        except TimeoutException:
            log.warning("  Timeout 30s")

        # Scroll progressivo para forçar lazy load
        for pct in [0.25, 0.5, 0.75, 1.0]:
            driver.execute_script(
                f"window.scrollTo(0, document.body.scrollHeight * {pct});")
            time.sleep(1)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(6)  # espera extra para conteúdo dinâmico

        links = driver.find_elements(By.TAG_NAME, "a")
        log.info(f"  Links na página: {len(links)}")

        for link in links:
            try:
                href  = link.get_attribute("href") or ""
                texto = link.text or ""
                aria  = link.get_attribute("aria-label") or ""
                title = link.get_attribute("title") or ""
                tc    = f"{href} {texto} {aria} {title}"
            except Exception:
                continue

            hl = _norm(href)
            # Aceita PDF direto OU botões de download
            eh_pdf = (".pdf" in hl or "download" in hl or
                      "arquivo" in hl or "getfile" in hl or
                      "attachment" in hl)
            if not eh_pdf:
                continue

            # Filtro de ano
            anos = re.findall(r"20\d{2}", tc)
            if anos and str(ano) not in anos:
                continue

            # Filtro de trimestre
            tri = _detectar_tri(tc)
            if tri and tri != trimestre:
                continue

            url_abs = href if href.startswith("http") else urljoin(url_base, href)

            for tipo, kws in KEYWORDS.items():
                if _contem(tc, kws):
                    if url_abs not in candidatos[tipo]:
                        candidatos[tipo].append(url_abs)
                    break
            else:
                if url_abs not in candidatos["release"]:
                    candidatos["release"].append(url_abs)

    except WebDriverException as e:
        log.warning(f"  Erro Selenium: {e}")
    return candidatos


def _baixar_pdf(url_pdf, caminho):
    if caminho.exists():
        log.info(f"  ⚠ Já existe: {caminho.name}")
        return True
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        r = session.get(url_pdf, timeout=45, verify=False, allow_redirects=True)

        if r.status_code != 200:
            log.warning(f"  ✗ HTTP {r.status_code} — {url_pdf[:80]}")
            return False

        c = r.content
        tamanho = len(c)
        content_type = r.headers.get("Content-Type", "")
        log.info(f"  Debug: {tamanho} bytes | CT={content_type[:50]} | inicio={c[:8]}")

        # Verificação flexível: aceita qualquer uma das condições
        eh_pdf = (
            c[:4] == b"%PDF"                          # assinatura padrão
            or b"%PDF" in c[:100]                     # BOM antes da assinatura
            or "pdf" in content_type.lower()          # Content-Type correto
            or (tamanho > 10000                       # arquivo grande
                and b"<html" not in c[:500].lower()   # não é HTML
                and b"<!doctype" not in c[:500].lower())
        )

        if not eh_pdf:
            log.warning(f"  ✗ Não parece PDF: inicio={c[:20]}")
            return False

        if tamanho < 5000:
            log.warning(f"  ✗ Muito pequeno ({tamanho} bytes)")
            return False

        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_bytes(c)
        log.info(f"  ✓ {caminho.name} ({tamanho/1024:.1f} KB)")
        return True

    except Exception as e:
        log.warning(f"  ✗ Erro download: {e}")
        return False


def _processar_empresa(driver, ticker, url_ri, ano, trimestre):
    log.info(f"\n{'─'*58}")
    log.info(f"  {ticker} | {ano} {trimestre}")

    resultado = {"ticker": ticker, "ano": ano, "trimestre": trimestre,
                 "tipo_baixado": None, "arquivo": None,
                 "status": "sem_cobertura", "url_usada": url_ri}

    # Lista de URLs para tentar: principal + alternativas
    urls_tentar = [url_ri]
    for alt in URLS_ALTERNATIVAS.get(ticker, []):
        if alt not in urls_tentar:
            urls_tentar.append(alt)

    for url in urls_tentar:
        log.info(f"  URL: {url[:70]}...")
        candidatos = _extrair_links(driver, url, ano, trimestre)
        total = sum(len(v) for v in candidatos.values())
        log.info(f"  Candidatos: transcricao={len(candidatos['transcricao'])} "
                 f"release={len(candidatos['release'])} "
                 f"apresentacao={len(candidatos['apresentacao'])}")

        if total == 0:
            continue  # tenta próxima URL

        for tipo in ("transcricao", "release", "apresentacao"):
            for url_pdf in candidatos[tipo]:
                h    = _hash(url_pdf)
                nome = f"{ticker}_{ano}_{trimestre}_{tipo}_{h}.pdf"
                cam  = DIRETORIO_PDFS / ticker / str(ano) / trimestre / nome
                if _baixar_pdf(url_pdf, cam):
                    resultado["tipo_baixado"] = tipo
                    resultado["arquivo"]      = str(cam)
                    resultado["status"]       = "sucesso"
                    resultado["url_usada"]    = url
                    return resultado

        resultado["status"] = "falha_download"

    log.info(f"  ✗ Sem PDF em nenhuma URL testada")
    return resultado


def _perguntar():
    print("\n" + "─"*58)
    print("CONFIGURAÇÃO")
    print("─"*58)
    e = input("  Ano (2023/2024/2025) [2024]: ").strip()
    try:
        ano = int(e) if e else 2024
        if ano not in (2023,2024,2025): ano = 2024
    except: ano = 2024
    t = input("  Trimestre (1T/2T/3T/4T) [4T]: ").strip().upper()
    tri = t if t in ("1T","2T","3T","4T") else "4T"
    print(f"\n  ✓ ano={ano} | trimestre={tri}")
    if input("  Confirmar? (Enter=sim / n=ajustar): ").strip().lower() == "n":
        return _perguntar()
    return {"ano": ano, "trimestre": tri}


def main():
    print("="*58)
    print("SCRAPER EARNINGS CALLS v2 — TOP 15 IBrX-100")
    print("="*58)

    if not CHROMEDRIVER_PATH.exists():
        print(f"ChromeDriver não encontrado: {CHROMEDRIVER_PATH}")
        return

    csv_path = input("\nCSV (url + ticker) [Enter = top15_ri.csv]: ").strip()
    if not csv_path:
        csv_path = str(RAIZ / "data" / "top15_ri.csv")
    if not os.path.exists(csv_path):
        print(f"Não encontrado: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if not {"url","ticker"}.issubset(df.columns):
        print("CSV precisa de colunas: url, ticker")
        return

    print(f"✓ {len(df)} empresas carregadas")
    params = _perguntar()
    ano, trimestre = params["ano"], params["trimestre"]

    DIRETORIO_PDFS.mkdir(parents=True, exist_ok=True)
    DIRETORIO_LOGS.mkdir(parents=True, exist_ok=True)

    log.info("Iniciando Chrome headless...")
    try:
        driver = _criar_driver()
    except Exception as e:
        print(f"Erro ChromeDriver: {e}")
        return

    resultados = []
    try:
        for _, row in df.iterrows():
            res = _processar_empresa(
                driver, str(row["ticker"]).strip().upper(),
                str(row["url"]).strip(), ano, trimestre)
            resultados.append(res)
            time.sleep(2)
    except KeyboardInterrupt:
        log.info("Interrompido.")
    finally:
        driver.quit()
        log.info("Chrome encerrado.")

    df_res = pd.DataFrame(resultados)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rel = DIRETORIO_LOGS / f"scraper_{ano}_{trimestre}_{ts}.csv"
    df_res.to_csv(rel, index=False, encoding="utf-8-sig")

    print(f"\n{'='*58}")
    print("RESUMO")
    print(f"{'='*58}")
    for status, label in [("sucesso","✓ Sucesso"),("sem_cobertura","✗ Sem cobertura"),
                          ("falha_download","! Falha download")]:
        n = (df_res["status"]==status).sum()
        print(f"  {label}: {n}")
    if (df_res["status"]=="sucesso").sum() > 0:
        print("\n  Por tipo:")
        for t,c in df_res[df_res["status"]=="sucesso"]["tipo_baixado"].value_counts().items():
            print(f"    {t}: {c}")
    print(f"\nRelatório: {rel}")

if __name__ == "__main__":
    main()

    