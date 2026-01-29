
from bcb import sgs
print("\n--- Teste 2022 (Curto) ---")

try:
    print("Tentando Selic 2022 (int)...")
    df = sgs.get({'selic': 432}, start=2022)
    print("Sucesso Selic 2022 int!")
except Exception as e:
    print(f"Erro Selic 2022: {e}")

try:
    print("Tentando Selic '01/01/2022'...")
    df = sgs.get({'selic': 432}, start='01/01/2022')
    print("Sucesso Selic pt-br!")
except Exception as e:
    print(f"Erro Selic pt-br: {e}")
    
try:
    print("Tentando Misto 2022...")
    df = sgs.get({'selic': 432, 'ibc_br': 24363}, start='2022-01-01')
    print("Sucesso Misto 2022!")
except Exception as e:
    print(f"Erro Misto 2022: {e}")
