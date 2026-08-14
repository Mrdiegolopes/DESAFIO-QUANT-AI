"""
Script de Download de PDFs com Links Fornecidos
Você cola os links dos PDFs e o script baixa automaticamente
Autor: Assistente
Data: 2026
"""

import requests
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json
import pandas as pd

# Configurações
DIRETORIO_PROJETO = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd()
DIRETORIO_PDFS = DIRETORIO_PROJETO / "pdfs_coletados"

class DownloaderLinks:
    """
    Downloader de PDFs a partir de links fornecidos manualmente
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
        }
        
        # Cria diretório de saída
        DIRETORIO_PDFS.mkdir(parents=True, exist_ok=True)
        
    def baixar_pdf_unico(self, url: str, ticker: str, ano: int, trimestre: str = 'NA', tipo: str = 'documento'):
        """
        Baixa um único PDF a partir de link fornecido
        
        Args:
            url: URL direta do PDF
            ticker: Código da ação (ex: ABEV3)
            ano: Ano do documento (ex: 2023)
            trimestre: Trimestre (ex: 4T, 3T, etc.)
            tipo: Tipo do documento (transcricao, apresentacao, release, etc.)
        """
        print(f"\n  Baixando PDF:")
        print(f"    Ticker: {ticker}")
        print(f"    Ano: {ano}")
        print(f"    Trimestre: {trimestre}")
        print(f"    Tipo: {tipo}")
        print(f"    URL: {url[:100]}...")
        
        # Cria diretório
        diretorio = DIRETORIO_PDFS / ticker / str(ano) / trimestre
        diretorio.mkdir(parents=True, exist_ok=True)
        
        # Nome do arquivo
        nome_arquivo = f"{ticker}_{ano}_{trimestre}_{tipo}.pdf"
        caminho_completo = diretorio / nome_arquivo
        
        # Verifica se já existe
        if caminho_completo.exists():
            print(f"    ⚠ Já existe: {caminho_completo.name}")
            return str(caminho_completo)
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 1000:
                with open(caminho_completo, 'wb') as f:
                    f.write(response.content)
                    
                tamanho = len(response.content) / 1024  # KB
                print(f"    ✓ Sucesso! ({tamanho:.1f} KB)")
                print(f"    Salvo em: {caminho_completo}")
                return str(caminho_completo)
            else:
                print(f"    ✗ Falha (status: {response.status_code})")
                return None
                
        except Exception as e:
            print(f"    ✗ Erro: {e}")
            return None
            
    def baixar_lista_pdfs(self, lista_pdfs: List[Dict]):
        """
        Baixa múltiplos PDFs de uma lista
        
        Args:
            lista_pdfs: Lista de dicionários com as chaves:
                - url: URL do PDF
                - ticker: Código da ação
                - ano: Ano
                - trimestre: Trimestre (opcional)
                - tipo: Tipo do documento (opcional)
        """
        print(f"\n{'='*60}")
        print(f"BAIXANDO {len(lista_pdfs)} PDFs")
        print(f"{'='*60}")
        
        sucessos = 0
        falhas = 0
        
        for pdf in lista_pdfs:
            url = pdf.get('url', '')
            ticker = pdf.get('ticker', 'EMPRESA')
            ano = pdf.get('ano', datetime.now().year)
            trimestre = pdf.get('trimestre', 'NA')
            tipo = pdf.get('tipo', 'documento')
            
            resultado = self.baixar_pdf_unico(url, ticker, ano, trimestre, tipo)
            
            if resultado:
                sucessos += 1
            else:
                falhas += 1
                
            time.sleep(1)  # Delay entre downloads
            
        print(f"\n{'='*60}")
        print(f"RESUMO")
        print(f"{'='*60}")
        print(f"Sucessos: {sucessos}")
        print(f"Falhas: {falhas}")
        print(f"Total: {sucessos + falhas}")
        print(f"{'='*60}")
        
    def baixar_pdfs_empresa(self, ticker: str, links_por_ano: Dict[int, List[Dict]]):
        """
        Baixa PDFs de uma empresa organizados por ano
        
        Args:
            ticker: Código da ação
            links_por_ano: Dicionário {ano: [{url, trimestre, tipo}, ...]}
        """
        print(f"\n{'='*60}")
        print(f"BAIXANDO PDFs DE {ticker}")
        print(f"{'='*60}")
        
        for ano, links in links_por_ano.items():
            print(f"\n  Ano {ano}: {len(links)} PDFs")
            
            for link_info in links:
                url = link_info.get('url', '')
                trimestre = link_info.get('trimestre', 'NA')
                tipo = link_info.get('tipo', 'documento')
                
                self.baixar_pdf_unico(url, ticker, ano, trimestre, tipo)
                time.sleep(1)

def receber_links_manual():
    """
    Modo interativo para receber links manualmente
    """
    print("INSERIR LINKS MANUALMENTE")
    print("\nInstruções:")
    print("1. Cole o link do PDF")
    print("2. Informe o ticker (ex: ABEV3)")
    print("3. Informe o ano (ex: 2023)")
    print("4. Informe o trimestre (ex: 4T) ou Enter para pular")
    print("5. Informe o tipo (transcricao, apresentacao, release) ou Enter")
    print("6. Digite 'sair' para terminar\n")
    
    downloader = DownloaderLinks()
    pdfs_baixados = []
    
    while True:
        print("\n" + "-"*60)
        url = input("URL do PDF (ou 'sair'): ").strip()
        
        if url.lower() == 'sair':
            break
            
        if not url:
            print("URL vazia, tente novamente.")
            continue
            
        ticker = input("Ticker (ex: ABEV3): ").strip().upper()
        
        if not ticker:
            ticker = "EMPRESA"
            
        try:
            ano = int(input("Ano (ex: 2023): ").strip())
        except:
            ano = datetime.now().year
            
        trimestre = input("Trimestre (1T, 2T, 3T, 4T ou Enter): ").strip().upper()
        if not trimestre:
            trimestre = 'NA'
            
        tipo = input("Tipo (transcricao, apresentacao, release ou Enter): ").strip().lower()
        if not tipo:
            tipo = 'documento'
            
        resultado = downloader.baixar_pdf_unico(url, ticker, ano, trimestre, tipo)
        
        if resultado:
            pdfs_baixados.append({
                'url': url,
                'ticker': ticker,
                'ano': ano,
                'trimestre': trimestre,
                'tipo': tipo,
                'arquivo': resultado
            })
            
    return pdfs_baixados

def receber_links_arquivo(arquivo: str):
    """
    Recebe links de um arquivo CSV ou JSON
    """
    downloader = DownloaderLinks()
    
    if arquivo.endswith('.csv'):
        df = pd.read_csv(arquivo)
        
        # Verifica colunas necessárias
        colunas_necessarias = ['url', 'ticker']
        if not all(col in df.columns for col in colunas_necessarias):
            print(f"Erro: O arquivo CSV precisa ter as colunas: {colunas_necessarias}")
            return
            
        # Converte para lista de dicionários
        lista_pdfs = []
        for _, row in df.iterrows():
            pdf = {
                'url': row['url'],
                'ticker': row['ticker'],
                'ano': row.get('ano', datetime.now().year),
                'trimestre': row.get('trimestre', 'NA'),
                'tipo': row.get('tipo', 'documento')
            }
            lista_pdfs.append(pdf)
            
        downloader.baixar_lista_pdfs(lista_pdfs)
        
    elif arquivo.endswith('.json'):
        with open(arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            
        if isinstance(dados, list):
            downloader.baixar_lista_pdfs(dados)
        else:
            print("Erro: JSON deve ser uma lista de objetos")
            
    else:
        print("Formato não suportado. Use CSV ou JSON.")

def main():
    """
    Função principal
    """
    print("="*60)
    print("DOWNLOADER DE PDFs COM LINKS FORNECIDOS")
    print("="*60)
    print(f"Diretório de saída: {DIRETORIO_PDFS}")
    print("="*60)
    print("""
Escolha o modo:
1. Inserir links manualmente (interativo)
2. Carregar links de arquivo CSV/JSON
3. Exemplo com links pré-definidos
0. Sair
    """)
    
    try:
        opcao = input("Opção: ").strip()
    except:
        return
        
    downloader = DownloaderLinks()
    
    if opcao == '1':
        pdfs = receber_links_manual()
        
        if pdfs:
            # Salva registro
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            df = pd.DataFrame(pdfs)
            df.to_csv(f'registro_downloads_{timestamp}.csv', index=False, encoding='utf-8-sig')
            print(f"\nRegistro salvo em: registro_downloads_{timestamp}.csv")
            
    elif opcao == '2':
        arquivo = input("Caminho do arquivo (CSV ou JSON): ").strip()
        
        if arquivo and os.path.exists(arquivo):
            receber_links_arquivo(arquivo)
        else:
            print("Arquivo não encontrado!")
            
    elif opcao == '3':
        # Exemplo com links pré-definidos
        exemplos = [
            {
                'url': 'https://exemplo.com/ambev_4T23_apresentacao.pdf',
                'ticker': 'ABEV3',
                'ano': 2023,
                'trimestre': '4T',
                'tipo': 'apresentacao'
            },
            {
                'url': 'https://exemplo.com/ambev_4T23_transcricao.pdf',
                'ticker': 'ABEV3',
                'ano': 2023,
                'trimestre': '4T',
                'tipo': 'transcricao'
            },
        ]
        
        print("\nExemplos de links (substitua pelos reais):")
        for ex in exemplos:
            print(f"  {ex['ticker']} {ex['ano']} {ex['trimestre']} - {ex['url']}")
            
        print("\nUse o modo 1 para inserir seus links reais.")
        
    elif opcao == '0':
        print("Saindo...")
        return
        
    print("\nProcesso concluído!")

if __name__ == "__main__":
    main() 