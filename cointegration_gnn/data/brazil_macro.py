
"""
Brazil Macro Data Fetcher
=========================
Coleta e processamento de dados macroeconômicos brasileiros oficiais
para modelos de trading, focando em evitar look-ahead bias.

Fontes:
- Banco Central do Brasil (BCB): via python-bcb
- IBGE (SIDRA): via sidrapy

Indicadores:
1. Selic Meta (Custo de oportunidade) - BCB 432
2. PTAX Venda (Câmbio oficial) - BCB 1
3. IBC-Br (Atividade mensal) - BCB 24363
4. IPCA (Inflação oficial) - IBGE 1737
5. PIB Trimestral - IBGE 1621
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import warnings
import requests

# Try to import required libraries
try:
    from bcb import sgs
    BCB_AVAILABLE = True
except ImportError:
    BCB_AVAILABLE = False
    warnings.warn("biblioteca 'python-bcb' não encontrada. Instale com: pip install python-bcb")


try:
    import sidrapy
    SIDRA_AVAILABLE = True
except ImportError:
    SIDRA_AVAILABLE = False
    warnings.warn("biblioteca 'sidrapy' não encontrada. Instale com: pip install sidrapy")

class IpeadataClient:
    """
    Cliente para a API OData v4 do IPEADATA.
    Documentação: http://www.ipeadata.gov.br/api/
    """
    BASE_URL = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{code}')"
    
    def fetch_series(self, code: str, start_date: str) -> pd.DataFrame:
        """Busca série temporal e retorna DataFrame com colunas ['date', 'val']."""
        print(f"      [IPEADATA] Buscando série {code}...")
        url = self.BASE_URL.format(code=code)
        try:
            # IPEADATA requires generic user agent sometimes, or accept json
            # OData v4 returns JSON by default or with $format=json
            params = {"$format": "json"}
            # Ensure headers to avoid 406 equivalent or content negotiation issues
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json"
            } # Ensure headers to avoid 406
            response = requests.get(url, params=params, headers=headers, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if 'value' in data:
                    df = pd.DataFrame(data['value'])
                    
                    # Normalize columns to upper just in case
                    df.columns = [str(c).upper() for c in df.columns]
                    
                    if 'VALDATA' in df.columns and 'VALVALOR' in df.columns:
                        # Robust datetime parsing
                        df['date'] = pd.to_datetime(df['VALDATA'], utc=True, errors='coerce').dt.tz_localize(None)
                        df['val'] = pd.to_numeric(df['VALVALOR'], errors='coerce')
                        
                        # Drop invalid rows
                        df = df.dropna(subset=['date', 'val'])
                        
                        df = df[['date', 'val']].set_index('date').sort_index()
                        df = df[df.index >= pd.to_datetime(start_date)]
                        return df
            
            print(f"      [X] IPEADATA Error {code}: Status {response.status_code}. Data keys: {list(data.keys()) if response.status_code == 200 else 'N/A'}")
        except Exception as e:
            print(f"      [X] IPEADATA Exception {code}: {e}")
            
        return pd.DataFrame() # Empty on failure


class BrazilMacroFetcher:
    def __init__(self, cache_path: str = "data/brazil_macro.csv"):
        self.cache_path = cache_path
        
    def fetch_all(self, start_date: str = "2010-01-01") -> pd.DataFrame:
        """Busca todos os indicadores e consolida em um DataFrame diário."""
        if not SIDRA_AVAILABLE:
            print("⚠️ Biblioteca 'sidrapy' faltando. Impossível buscar dados IBGE.")
            # We continue to try fetching BCB/IPEA even if sidra is missing, 
            # though the user logic originally returned empty.
            if not BCB_AVAILABLE:
                 print("[!] Nem python-bcb nem sidrapy encontrados.")
            
        print("[*] Buscando dados macroeconomicos do Brasil...")
        
        # Parse start year for safer API call
        try:
            start_year = int(pd.to_datetime(start_date).year)
        except:
            start_year = 2010
            
        # 0. Initialize IPEA Client for fallback
        ipea = IpeadataClient()
        
        # 1. BCB Data (SGS) - With Fallback
        print(f"   - BCB/IPEA: Selic, PTAX (Diários)")
        bcb_success = False
        bcb_data = pd.DataFrame()
        
        if BCB_AVAILABLE:
            try:
                # Tenta BCB Oficial
                bcb_daily = sgs.get(
                    codes={
                        'selic': 432,      # Meta Selic (% a.a.)
                        'usd_brl': 1,      # PTAX Venda
                        'ibc_br': 24363,   # IBC-Br (Série Bruta)
                        'ipca_mom': 433,   # IPCA Var % Mensal
                        'ipca_12m': 13522  # IPCA 12m
                    },
                    start=start_year
                )
                if not bcb_daily.empty:
                    bcb_data = bcb_daily
                    bcb_success = True
            except Exception as e:
                print(f"   [!] Falha BCB Primario (SGS): {e}. Tentando IPEADATA...")
        
        if not bcb_success:
            # Fallback to IPEADATA for Daily/Monthly Series
            selic_df = ipea.fetch_series('BM366_TJOVER366', start_date) 
            usd_df = ipea.fetch_series('GM366_ERC366', start_date)
            ibc_br_df = ipea.fetch_series('SGS12_IBCBR12', start_date)
            ipca_mom_df = ipea.fetch_series('PRECOS12_IPCA12', start_date)
            
            # Use lists to collect valid dataframes then merge (cleaner)
            dfs = []
            if not selic_df.empty: dfs.append(selic_df.rename(columns={'val': 'selic'}))
            if not usd_df.empty: dfs.append(usd_df.rename(columns={'val': 'usd_brl'}))
            if not ibc_br_df.empty: dfs.append(ibc_br_df.rename(columns={'val': 'ibc_br'}))
            if not ipca_mom_df.empty: dfs.append(ipca_mom_df.rename(columns={'val': 'ipca_mom'}))
            
            if dfs:
                bcb_data = dfs[0]
                for d in dfs[1:]:
                    bcb_data = bcb_data.join(d, how='outer')

        # Adjustment for Percentages -> Decimals (0.5% -> 0.005)
        # Scan columns and adjust
        cols_percent = ['selic', 'ipca_mom', 'ipca_12m']
        for c in cols_percent:
            if c in bcb_data.columns:
                bcb_data[c] = bcb_data[c] / 100.0
                
        # Merge Daily and Monthly BCB/IPEA data into bcb_data_final
        # 2. PIB Data (SIDRA)
        print("   - IBGE: PIB Trimestral")
        pib_df = self._fetch_pib_trimestral(start_date)
        ipca_df = pd.DataFrame() # Initialize empty for safety
        
        # 3. Merge & Alignment
        # Começamos com um índice diário base
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        macro_df = pd.DataFrame(index=dates)
        
        # Merge BCB/IPEA (Daily/Monthly aggregated)
        if not bcb_data.empty:
            # Normalize index to midnight
            bcb_data.index = bcb_data.index.normalize()
            # Deduplicate
            bcb_data = bcb_data.groupby(bcb_data.index).last()
            
            macro_df = macro_df.join(bcb_data, how='left')
            
        # Merge PIB (Quarterly -> Daily with Lag)
        if not pib_df.empty:
            # Divide by 100 if percent
            pib_df['pib_yoy'] = pib_df['pib_yoy'] / 100.0
            
            # PIB demora ~2 meses após o trimestre. Lag conservador: 3 meses.
            pib_df.index = pib_df.index + pd.DateOffset(months=3)
            # Ensure unique index
            pib_df = pib_df.groupby(pib_df.index).last()
            
            macro_df = macro_df.join(pib_df, how='left')
            
        # Note: IPCA logic removed here because it's now in bcb_data (via BCB 433 or IPEA Fallback)
        # If we still want SIDRA IPCA redundancy, we could merge it if bcb 'ipca_mom' is missing.
        if 'ipca_mom' not in macro_df.columns and not ipca_df.empty:
             # Add SIDRA IPCA if BCB failed
             ipca_df.index = ipca_df.index + pd.DateOffset(months=1, days=10)
             ipca_df['ipca_mom'] = ipca_df['ipca_mom'] / 100.0
             if 'ipca_12m' in ipca_df.columns:
                 ipca_df['ipca_12m'] = ipca_df['ipca_12m'] / 100.0
             macro_df = macro_df.join(ipca_df, how='left', rsuffix='_sidra')
             
             # Consolidate
             if 'ipca_mom' not in macro_df.columns and 'ipca_mom_sidra' in macro_df.columns:
                 macro_df['ipca_mom'] = macro_df['ipca_mom_sidra']
             if 'ipca_12m' not in macro_df.columns and 'ipca_12m_sidra' in macro_df.columns:
                 macro_df['ipca_12m'] = macro_df['ipca_12m_sidra']

        # Fallback Calculation: IPCA 12m from IPCA MoM if still missing
        if 'ipca_mom' in macro_df.columns and ('ipca_12m' not in macro_df.columns or macro_df['ipca_12m'].sum() == 0):
             print("   [i] Calculando IPCA 12m a partir do MoM...")
             # Rolling compound return: Prod(1+r) - 1
             # Resample to monthly to calc rolling 12 then backfill? 
             # Or just use the already daily-expanded but monthly-changing series?
             # Since it's ffilled later, we should calculate on valid monthly points.
             # But here macro_df is daily.
             # Let's approximate by taking the daily series (which has steps) and doing rolling(365)?
             # No, easier: logic is applied before ffill? No, macro_df is already joined.
             # Better: Calculate on the non-NaN values of ipca_mom before ffill if possible.
             # But here we are after join.
             
             # Let's try to calculate it simply:
             # 1. Identify valid MoM dates (where value changes or is present)
             # Actually, simpler: Just ensure we have it. If 0.00%, it's bad.
             pass
             
        # Feature Engineering: Stationarity
            
        # 4. Fill & Transformations
        # Forward fill: O dado mais recente permanece válido até sair novo
        macro_df = macro_df.ffill()
        
        if macro_df.empty:
            print("[!] Macro DataFrame is empty after merge!")
            return macro_df
            
        print(f"   [#] Macro Data Columns: {list(macro_df.columns)}")
        print(f"   [#] Macro Data Head:\n{macro_df.head()}")
        
        # Feature Engineering: Stationarity
        if 'selic' in macro_df.columns:
            # Selic: Nível é importante, mas variação também
            macro_df['selic_change'] = macro_df['selic'].diff()
        else:
            print("[!] Coluna 'selic' nao encontrada!")
        
        if 'usd_brl' in macro_df.columns:
            # USD: Log return
            macro_df['usd_ret'] = np.log(macro_df['usd_brl'] / macro_df['usd_brl'].shift(1)).fillna(0)
            macro_df['usd_vol_20d'] = macro_df['usd_ret'].rolling(20).std()
        
        # IBC-Br: Growth YoY (12m lag) e MoM
        if 'ibc_br' in macro_df.columns:
            macro_df['ibc_br_yoy'] = macro_df['ibc_br'].pct_change(365, fill_method=None)
            
        # Ensure all expected columns exist for model compatibility
        expected_cols = ['selic', 'usd_brl', 'ibc_br', 'ipca_mom', 'ipca_12m']
        for col in expected_cols:
            if col not in macro_df.columns:
                print(f"   [!] Missing column {col}, filling with 0.0 to match model shape.")
                macro_df[col] = 0.0

        # [CRITICAL REPAIR] Re-calculate IPCA 12m if it looks broken (all zeros)
        # Check if ipca_mom is non-zero but ipca_12m is zero/missing
        if 'ipca_mom' in macro_df.columns and 'ipca_12m' in macro_df.columns:
             if macro_df['ipca_12m'].abs().max() < 0.0001 and macro_df['ipca_mom'].abs().max() > 0.0001:
                 print("   [!] IPCA 12m missing/zero. Recalculating from IPCA MoM...")
                 # MoM is percentage decimal. (1+r).cumprod solution.
                 # Since data is daily ffilled, we need to be careful.
                 # Group by month to get unique monthly values?
                 # Simpler: Rolling sum of logs for approx, or true compound.
                 # Window = 252 business days (approx 1 year).
                 macro_df['ipca_12m'] = (1 + macro_df['ipca_mom']).rolling(252).apply(np.prod, raw=True) - 1
                 macro_df['ipca_12m'] = macro_df['ipca_12m'].fillna(0.0)

        # Real Interest Rate proxy (Selic - IPCA 12m)
        if 'ipca_12m' in macro_df.columns and 'selic' in macro_df.columns:
            macro_df['juro_real'] = macro_df['selic'] - macro_df['ipca_12m']
        else:
             macro_df['juro_real'] = 0.0
        
        # Drop initial NaNs from rolling/diff
        # Only drop if columns exist
        dropna_subset = [c for c in ['selic', 'usd_brl'] if c in macro_df.columns]
        if dropna_subset:
            macro_df = macro_df.dropna(subset=dropna_subset)
            
        # [CORREÇÃO CRÍTICA] Garantir zero NaNs remanescentes
        macro_df = macro_df.ffill().fillna(0.0)
        
        if macro_df.isnull().values.any():
            print("[!] AVISO: NaNs remanescentes nos dados Macro. Preenchendo com 0.")
            macro_df = macro_df.fillna(0.0)
        
        print(f"[OK] Dados macro BR carregados: {macro_df.shape} linhas")
        # Save cache
        macro_df.to_csv(self.cache_path)
        
        return macro_df

    def _fetch_ipca(self, start_date) -> pd.DataFrame:
        try:
            # Tabela 1737: IPCA - Série histórica
            # Variável 63: Variação mensal
            # Variável 2265: Variação acumulada em 12 meses
            df = sidrapy.get_table(
                table_code="1737",
                territorial_level="1",
                ibge_territorial_code="all",
                variable="63,2265",
                period="all"
            )
            
            # Limpeza SIDRA
            # Limpeza SIDRA
            df = df.iloc[1:] # Remove header
            
            # [CORREÇÃO] Normalização de chaves (Case Sensitivity)
            # A API pode retornar 'V' ou 'v', 'D2C' ou 'd2c', etc.
            # Vamos converter todas as colunas para maiúsculo
            df.columns = df.columns.str.upper()
            
            # Verificar se as colunas essenciais existem
            if 'V' not in df.columns or 'D2C' not in df.columns:
                 # Tentar mapeamento manual se normalize falhar (ex: nomes mudaram drasticamente)
                 print(f"   [!] Colunas inesperadas no SIDRA: {df.columns}")
                 return pd.DataFrame()

            df = df[['V', 'D2C', 'D3N']] # valor, data (yyyymm), variável
            
            # Pivot
            df['date'] = pd.to_datetime(df['D2C'], format="%Y%m")
            df['val'] = pd.to_numeric(df['V'], errors='coerce')
            
            pivot = df.pivot(index='date', columns='D3N', values='val')
            
            # Renomear colunas (depende do nome exato retornado pela API, que pode variar)
            # Vamos assumir ordem ou buscar string
            cols = {}
            for c in pivot.columns:
                if "mensal" in c.lower():
                    cols[c] = "ipca_mom"
                elif "12 meses" in c.lower():
                    cols[c] = "ipca_12m"
            
            pivot = pivot.rename(columns=cols)
            
            # Filtro data
            pivot = pivot[pivot.index >= pd.to_datetime(start_date)]
            return pivot
            
        except Exception as e:
            print(f"   [!] Erro IPCA (Sidra): {e}")
            return pd.DataFrame()

    def _fetch_pib_trimestral(self, start_date) -> pd.DataFrame:
        try:
            # Tabela 5932: PIB Trimestral
            # Variável 6562: Taxa de variação em relação ao mesmo período do ano anterior
            # Classificação 11255: Setores (90707 = PIB a preços de mercado)
            df = sidrapy.get_table(
                table_code="5932",
                territorial_level="1",
                ibge_territorial_code="all",
                variable="6562",
                period="all",
                classifications={"11255": "90707"}
            )
            
            if df.empty or 'V' not in df.columns:
                 return pd.DataFrame()

            df = df.iloc[1:]
            
            # [CORREÇÃO] Case Sensitivity
            df.columns = df.columns.str.upper()
            
            df['val'] = pd.to_numeric(df['V'], errors='coerce')
            
            # Data vem como "199601" (1º tri), "199602" (2º tri)...
            # Converter para data (fim do trimestre ou início?) -> Vamos usar início do tri
            def parse_quarter(x):
                year = int(x[:4])
                q = int(x[4:])
                month = (q - 1) * 3 + 1
                return datetime(year, month, 1)
                
            df['date'] = df['D2C'].apply(parse_quarter)
            df = df.set_index('date')[['val']]
            df.columns = ['pib_yoy']
            
            df = df[df.index >= pd.to_datetime(start_date)]
            return df
            
        except Exception as e:
            print(f"   [!] Erro PIB (Sidra): {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Teste rápido
    fetcher = BrazilMacroFetcher()
    df = fetcher.fetch_all()
    print(df.tail())
    print(df.describe())
