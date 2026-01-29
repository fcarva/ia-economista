
print("1. Start")
import torch
print("2. Torch Imported")
import torch_geometric
print("3. PyG Imported")
try:
    import pytorch_lightning
    print("4. PL Imported")
except ImportError:
    print("4. PL Import Failed")
except Exception as e:
    print(f"4. PL Import Error: {e}")

print("5. Done")
