import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pdfplumber
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class EarningsPipeline:
    def __init__(self, output_dir: str = "data/ibrx100_earnings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def fetch_ri_pdf_url(self, ticker: str, quarter: str, year: int) -> Optional[str]:
        """
        Mecanismo de busca automatizado para encontrar a URL do PDF de transcrição.
        Tenta buscar via padrões conhecidos de portais de RI (MZ Group, RIWeb, etc.).
        """
        logging.info(f"Buscando URL do PDF de teleconferência para {ticker} ({quarter}{year})...")
        
        # Padrão genérico de busca/raspagem (pode ser expandido por provedor de RI)
        search_query = f"{ticker} transcricao teleconferencia {quarter} {year} pdf"
        
        # Exemplo de consulta via endpoint de busca ou mapeamento direto de RI
        # Para produção, mantemos um dicionário/banco com os portais base de RI das 100 empresas
        # Caso tenha o link direto, ele é retornado aqui.
        return None

    def extract_raw_text(self, pdf_path: Path) -> str:
        """
        Extrai o texto com suporte a layouts de coluna única e coluna dupla.
        """
        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Tenta extração padrão
                text = page.extract_text(layout=False)
                
                # Se o texto parecer desalinhado ou muito curto (comum em colunas duplas), aciona layout=True
                if not text or len(text.strip()) < 100:
                    text = page.extract_text(layout=True)
                    
                if text:
                    # Limpeza de ruídos e cabeçalhos
                    text = re.sub(r'(?i)página\s+\d+\s+de\s+\d+', '', text)
                    text = re.sub(r'(?i)page\s+\d+\s+of\s+\d+', '', text)
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                    full_text.append(text)
                    
        return "\n\n".join(full_text)

    def parse_transcript_to_json(
        self, raw_text: str, ticker: str, quarter: str, year: int, source_url: str
    ) -> Dict[str, Any]:
        """Converte a transcrição bruta em blocos estruturados por orador e seção."""
        
        qa_pattern = r'(sessão de perguntas e respostas|question & answer session|perguntas e respostas|q&a)'
        parts = re.split(qa_pattern, raw_text, flags=re.IGNORECASE)

        prepared_text = parts[0]
        qa_text = parts[2] if len(parts) > 2 else ""

        # Captura o orador no formato "Nome do Executivo (Cargo):" ou "Nome do Analista - Instituição:"
        speaker_pattern = r'([A-Z][a-zA-Zà-úÀ-Ú\s]+)\s*(?:\(([^)]+)\)|-\s*([^:]+))?\s*:'

        def extract_speech_blocks(segment: str, section_name: str) -> List[Dict[str, Any]]:
            blocks = []
            matches = list(re.finditer(speaker_pattern, segment))
            
            for i in range(len(matches)):
                speaker = matches[i].group(1).strip()
                meta = matches[i].group(2) or matches[i].group(3) or "Não especificado"
                
                start_idx = matches[i].end()
                end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(segment)
                
                speech = segment[start_idx:end_idx].strip()
                speech = re.sub(r'\s+', ' ', speech)
                
                if len(speech) > 20:  # Filtra ruídos mínimos
                    blocks.append({
                        "chunk_id": f"{ticker}_{year}_{quarter}_{section_name}_{i+1:03d}",
                        "speaker": speaker,
                        "role_or_affiliation": meta.strip(),
                        "text": speech,
                        "token_estimate": len(speech.split())  # Útil para janelas do RAG
                    })
            return blocks

        return {
            "doc_id": f"{ticker.upper()}_{year}_{quarter.upper()}_TRANSCRIPT",
            "metadata": {
                "ticker": ticker.upper(),
                "quarter": quarter.upper(),
                "year": year,
                "source_url": source_url,
                "processed_at": datetime.utcnow().isoformat() + "Z"
            },
            "sections": {
                "prepared_remarks": extract_speech_blocks(prepared_text, "remarks"),
                "qa_session": extract_speech_blocks(qa_text, "qa")
            }
        }

    def process_item(self, ticker: str, quarter: str, year: int, pdf_url: str):
        """Executa o pipeline completo: Download -> Extração -> JSON."""
        file_prefix = f"{ticker.upper()}_{year}_{quarter.upper()}"
        pdf_path = self.output_dir / f"{file_prefix}.pdf"
        json_path = self.output_dir / f"{file_prefix}.json"

        # Step 1: Download
        try:
            res = requests.get(pdf_url, headers=self.headers, timeout=30)
            res.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(res.content)
            logging.info(f"[{ticker}] PDF salvo com sucesso.")
        except Exception as e:
            logging.error(f"[{ticker}] Erro no download: {e}")
            return

        # Step 2: Parsing & Extração
        raw_text = self.extract_raw_text(pdf_path)
        structured_data = self.parse_transcript_to_json(
            raw_text=raw_text,
            ticker=ticker,
            quarter=quarter,
            year=year,
            source_url=pdf_url
        )

        # Step 3: Salvamento
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)

        logging.info(f"[{ticker}] JSON final gerado em: {json_path}")