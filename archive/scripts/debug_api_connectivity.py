import requests
import pandas as pd
import json
from datetime import datetime

# Configuração para parecer um navegador (evita bloqueio 403 em alguns casos)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/json"
}

def test_bcb_sgs(codigo_serie=432):
    """
    Testa a API do Banco Central (SGS).
    Padrão: 432 (Meta Selic).
    """
    # TENTATIVA HTTPS
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
    print(f"\n--- Testando BCB (Série {codigo_serie}) [HTTPS] ---")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                df = pd.DataFrame(data)
                print("✅ Sucesso! Dados recebidos.")
                print(f"Exemplo (últimos 3 registros):\n{df.tail(3)}")
                return True
            except json.JSONDecodeError:
                print("❌ Erro: A resposta não é um JSON válido.")
                print(f"Conteúdo bruto: {response.text[:200]}")
        else:
            print("❌ Falha na requisição.")
    except Exception as e:
        print(f"⚠️ Exceção crítica (Timeout ou Conexão): {e}")
    return False

def test_sidra_ibge():
    """
    Testa a API do IBGE (Sidra).
    Padrão: IPCA (Tabela 1737), Variação Mensal, Brasil, Últimos 3 meses.
    """
    # URL decodificada: Tabela 1737, Nível Territorial 1 (Brasil), Variável 63 (IPCA), Períodos (últimos 3)
    url = "https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/63/p/last%203/d/v63%202"
    print(f"\n--- Testando IBGE/Sidra (IPCA) ---")
    print(f"URL: {url}")
    
    try:
        # Sidra as vezes requer verificação SSL desativada em ambientes corporativos, mas vamos tentar com verify=True primeiro
        response = requests.get(url, headers=HEADERS, timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                # Sidra retorna o cabeçalho como primeiro elemento, as vezes confunde o Pandas
                if len(data) > 0:
                    print("✅ Sucesso! Dados recebidos.")
                    sample = data[1] if len(data)>1 else data[0]
                    print(f"Exemplo (Primeiro registro): {sample}")
                    print(f"Chaves disponíveis: {list(sample.keys())}")
                    return True
                else:
                    print("⚠️ Aviso: Retornou JSON vazio (lista vazia).")
            except json.JSONDecodeError:
                print("❌ Erro: A resposta não é um JSON válido.")
        else:
            print(f"❌ Falha. Motivo provável: {response.reason}")
    except Exception as e:
        print(f"⚠️ Exceção crítica (Timeout ou Conexão): {e}")
    return False

if __name__ == "__main__":
    print(f"Iniciando diagnóstico em: {datetime.now()}")
    bcb_ok = test_bcb_sgs()
    sidra_ok = test_sidra_ibge()
    
    print("\n" + "="*30)
    print("RESUMO DO DIAGNÓSTICO")
    print(f"BCB:   {'ONLINE ✅' if bcb_ok else 'OFFLINE ❌'}")
    print(f"SIDRA: {'ONLINE ✅' if sidra_ok else 'OFFLINE ❌'}")
    print("="*30)
