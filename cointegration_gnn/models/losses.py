
"""
Learning to Rank Losses for Quantitative Finance
================================================
Implementações de funções de perda focadas em ordenação (Ranking),
substituindo o MSE tradicional para estratégias Long-Short.

Algoritmos:
- ListMLE: Maximum Likelihood Estimation para permutações (Plackett-Luce).
- ListFold: Versão simétrica do ListMLE (Top-K e Bottom-K).

Referência:
"Learning to Rank for Long-Short Portfolio Construction"
"""

import torch
import torch.nn as nn
from torch import Tensor

class ListMLELoss(nn.Module):
    """ListMLE Loss: Maximiza a verossimilhança da permutação correta."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, scores: Tensor, target_returns: Tensor) -> Tensor:
        """
        Args:
            scores: (N,) Predicted scores (unbounded)
            target_returns: (N,) Actual returns
        """
        # 1. Obter a permutação correta (sort by return descending)
        # [Improvement] Add noise to break ties
        noise = torch.rand_like(target_returns) * 1e-6
        permutation = (target_returns + noise).argsort(descending=True)
        sorted_scores = scores[permutation]
        
        # 2. Estabilidade numérica (LogSumExp trick)
        # s_i - log(sum(exp(s_k))) para k >= i
        # Equivalente a: log( exp(s_i) / sum(exp(s_k)) )
        
        # Normalizar scores para evitar overflow no exp
        sorted_scores = sorted_scores - sorted_scores.max()
        
        # Calcular log_cumsum (cumulative logsumexp from end)
        n = len(sorted_scores)
        log_probs = torch.zeros(n, device=scores.device)
        
        # Implementação eficiente O(N) para logsumexp cumulativo é complexa em Torch puro sem loop.
        # Com loop é O(N^2) naive ou O(N) com scan. O loop reverso é aceitável para N ~ 100-500.
        
        # Vamos usar uma abordagem iterativa reversa para estabilidade
        current_log_sum_exp = -float('inf')
        
        loss_terms = []
        
        for i in range(n - 1, -1, -1):
            score = sorted_scores[i]
            # log_sum_exp(current, score)
            if current_log_sum_exp == -float('inf'):
                current_log_sum_exp = score
            else:
                current_log_sum_exp = torch.logaddexp(current_log_sum_exp, score)
                
            # Log Likelihood do item i ser o primeiro do subgrupo restante
            # log P(i | rest) = score_i - log_sum_exp_rest
            log_prob = score - current_log_sum_exp
            loss_terms.append(log_prob)
            
        # Loss = -Sum(log_probs)
        # loss_terms contém as log-probabilidades. Queremos maximizar a soma = minimizar o negativo.
        return -torch.stack(loss_terms).mean()


class ListFoldLoss(nn.Module):
    """
    ListFold: Symmetric ListMLE.
    Otimiza simetricamente o Topo (Long) e o Fundo (Short) da lista.
    Crucial para estratégias Market Neutral.
    """
    def __init__(self):
        super().__init__()
        self.listmle = ListMLELoss()
        
    def forward(self, scores: Tensor, target_returns: Tensor) -> Tensor:
        # 1. Forward Ranking (Melhores ativos no topo)
        loss_top = self.listmle(scores, target_returns)
        
        # 2. Backward Ranking (Piores ativos no topo)
        # Invertemos scores e targets. O "pior" retorno (-0.05) vira o maior (+0.05).
        # O menor score deve virar o maior score.
        loss_bottom = self.listmle(-scores, -target_returns)
        
        return (loss_top + loss_bottom) / 2
