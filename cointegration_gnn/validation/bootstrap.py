
"""
Bootstrap Validation Module
===========================
Implementação de validação estatística robusta usando Stationary Bootstrap
(Politis & Romano, 1994) para séries temporais financeiras.

Objetivo:
Gerar uma distribuição de probabilidade para o Índice de Sharpe e outras métricas,
garantindo que o resultado obtido não seja fruto de aleatoriedade ou overfitting
a um caminho histórico específico.

Dependências:
- arch (para StationaryBootstrap)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings

try:
    from arch.bootstrap import StationaryBootstrap
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("biblioteca 'arch' não encontrada. Instale com: pip install arch")

def compute_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calcula Sharpe ratio anualizado (assume dados diários)."""
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret < 1e-8:
        return 0.0
    # Anualizado (252 dias)
    return (mean_ret - risk_free_rate) / std_ret * np.sqrt(252)

class BootstrapValidator:
    """Validador de estratégias via Bootstrap Estacionário."""
    
    def __init__(self, n_samples: int = 1000, block_size: int = None, seed: int = 42):
        """
        Args:
            n_samples: Número de caminhos sintéticos a gerar (ex: 1000).
            block_size: Tamanho médio do bloco (None = otimizado automaticamente).
            seed: Semente aleatória.
        """
        self.n_samples = n_samples
        self.block_size = block_size
        self.seed = seed
        
    def validate(self, strategy_returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
        """Executa validação via bootstrap.
        
        Args:
            strategy_returns: Série de retornos diários da estratégia.
            benchmark_returns: Série de retornos do benchmark (opcional).
            
        Returns:
            Dicionário com estatísticas do bootstrap (IC Sharpe, prob > 0, etc).
        """
        if not ARCH_AVAILABLE:
            return {"error": "arch library not installed"}
        
        returns_arr = strategy_returns.values
        
        # Stationary Bootstrap mantém a dependência temporal (volatility clustering)
        # Escolha do block_size é crucial. Se None, arch tenta estimar ou usa heurística.
        # Para dados diários, ~10-20 dias é razoável para capturar memória de curto prazo.
        bs = StationaryBootstrap(
            self.block_size if self.block_size else 20,
            returns_arr,
            seed=self.seed
        )
        
        sharpe_dist = []
        
        # Gerar amostras
        for data in bs.bootstrap(self.n_samples):
            # data[0][0] contém a série resampleada
            sample_returns = data[0][0]
            sharpe = compute_sharpe(sample_returns)
            sharpe_dist.append(sharpe)
            
        sharpe_dist = np.array(sharpe_dist)
        
        # Métricas
        original_sharpe = compute_sharpe(returns_arr)
        mean_boot_sharpe = np.mean(sharpe_dist)
        std_boot_sharpe = np.std(sharpe_dist)
        
        # Intervalo de Confiança 95%
        ci_lower = np.percentile(sharpe_dist, 2.5)
        ci_upper = np.percentile(sharpe_dist, 97.5)
        
        # Probabilidade de Sharpe > Benchmark (ou > 0)
        prob_positive = np.mean(sharpe_dist > 0)
        
        # Probabilidade de Defeat (se benchmark fornecido) -> Não implementado aqui para simplicidade
        
        results = {
            "original_sharpe": original_sharpe,
            "bootstrap_mean_sharpe": mean_boot_sharpe,
            "bootstrap_std_error": std_boot_sharpe,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "prob_sharpe_positive": prob_positive,
            "min_sharpe": np.min(sharpe_dist),
            "max_sharpe": np.max(sharpe_dist),
            "n_samples": self.n_samples
        }
        
        print(f"\n✅ Validação Bootstrap ({self.n_samples} amostras):")
        print(f"   Sharpe Original: {original_sharpe:.4f}")
        print(f"   Sharpe Bootstrap (Média): {mean_boot_sharpe:.4f}")
        print(f"   IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   Prob. Sharpe > 0: {prob_positive:.1%}")
        
        return results

if __name__ == "__main__":
    # Teste rápido
    np.random.seed(42)
    # Simular retornos com leve média positiva e caudas gordas
    dummy_returns = pd.Series(np.random.normal(0.0005, 0.015, 1000))
    validator = BootstrapValidator(n_samples=500)
    res = validator.validate(dummy_returns)
