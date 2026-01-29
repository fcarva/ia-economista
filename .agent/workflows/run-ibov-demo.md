---
description: Como rodar o demo IBOVESPA e gerar visualizações
---

# Rodar Demo IBOVESPA

// turbo-all

1. Ativar ambiente virtual (se necessário):
```bash
.venv\Scripts\activate
```

2. Rodar demo completo:
```bash
python -m cointegration_gnn.demos.ibov_demo
```

3. Verificar outputs gerados:
```bash
ls outputs/
```

Os seguintes arquivos serão gerados:
- `ibov_cointegration_matrix.png` - Matriz de cointegração
- `ibov_causal_heatmap.png` - Heatmap Granger-Lasso
- `ibov_causal_network.png` - Rede causal

4. Para ver texto de resumo:
```bash
cat outputs/ibov_summary.txt
```
