"""
Monte Carlo Stress Testing (Track 4)
=====================================

Robustness validation for trading strategies using:
1. Maximum Entropy Bootstrap (Vinod, 2006)
2. Block Bootstrap for time series
3. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)

The goal is to test if the strategy's performance is statistically significant
after accounting for multiple testing bias and data snooping.

References:
    - Vinod (2006) "Maximum Entropy Bootstrap for Time Series"
    - Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
    - White (2000) "A Reality Check for Data Snooping"
"""

from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StressTestResult:
    """Results from Monte Carlo stress testing."""
    observed_sharpe: float
    mean_bootstrap_sharpe: float
    std_bootstrap_sharpe: float
    pvalue: float
    deflated_sharpe: float
    is_significant: bool
    confidence_level: float
    n_simulations: int
    bootstrap_sharpes: np.ndarray


class DeflatedSharpeRatio:
    """Calculate the Deflated Sharpe Ratio.
    
    Adjusts the Sharpe ratio for:
    1. Track record length
    2. Skewness and kurtosis
    3. Multiple testing (number of trials)
    
    Reference:
        Bailey & Lopez de Prado (2014)
        "The Deflated Sharpe Ratio: Correcting for Selection Bias,
        Backtest Overfitting, and Non-Normality"
    """
    
    @staticmethod
    def calculate(
        returns: np.ndarray,
        n_trials: int = 1,
        annualization_factor: float = 252.0,
    ) -> Tuple[float, float, float]:
        """Calculate deflated Sharpe ratio.
        
        Args:
            returns: Array of strategy returns.
            n_trials: Number of strategy variants tested.
            annualization_factor: Days per year for annualization.
            
        Returns:
            Tuple of (observed_sharpe, deflated_sharpe, p_value)
        """
        n = len(returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        # Observed Sharpe
        sr = (mean_ret / std_ret) * np.sqrt(annualization_factor)
        
        # Moments for adjustment
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
        
        # Expected maximum Sharpe under null (multiple testing)
        if n_trials > 1:
            # E[max(Z_1, ..., Z_n)] approximation
            expected_max = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/n_trials) + \
                          np.euler_gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))
        else:
            expected_max = 0
        
        # Variance of Sharpe estimate (Lo, 2002)
        # Var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / n
        var_sr = (1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / n
        std_sr = np.sqrt(var_sr)
        
        # Deflated Sharpe (test statistic)
        dsr = (sr - expected_max * std_sr) / std_sr if std_sr > 0 else 0
        
        # P-value under null
        pvalue = 1 - stats.norm.cdf(dsr)
        
        return sr, dsr, pvalue


class BlockBootstrap:
    """Block Bootstrap for time series data.
    
    Preserves autocorrelation structure by resampling blocks of data.
    """
    
    def __init__(
        self,
        block_size: int = 20,
        n_samples: int = 1000,
        random_state: Optional[int] = 42,
    ):
        """Initialize block bootstrap.
        
        Args:
            block_size: Size of contiguous blocks to resample.
            n_samples: Number of bootstrap samples.
            random_state: Random seed.
        """
        self.block_size = block_size
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)
    
    def resample(self, data: np.ndarray) -> np.ndarray:
        """Generate one bootstrap sample.
        
        Args:
            data: Original time series.
            
        Returns:
            Resampled time series of same length.
        """
        n = len(data)
        n_blocks = int(np.ceil(n / self.block_size))
        
        # Randomly select block starting points
        max_start = n - self.block_size
        if max_start <= 0:
            return data.copy()
        
        starts = self.rng.integers(0, max_start, size=n_blocks)
        
        # Concatenate blocks
        blocks = [data[s:s+self.block_size] for s in starts]
        resampled = np.concatenate(blocks)[:n]
        
        return resampled
    
    def bootstrap_statistic(
        self,
        data: np.ndarray,
        statistic_func: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """Compute bootstrap distribution of a statistic.
        
        Args:
            data: Original time series.
            statistic_func: Function that computes statistic from data.
            
        Returns:
            Array of bootstrap statistic values.
        """
        stats_array = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            boot_data = self.resample(data)
            stats_array[i] = statistic_func(boot_data)
        
        return stats_array


class MaxEntropyBootstrap:
    """Maximum Entropy Bootstrap for time series.
    
    Better preserves the dependence structure than block bootstrap.
    Uses a non-parametric, entropy-maximizing approach.
    
    Reference:
        Vinod (2006) "Maximum Entropy Bootstrap for Time Series"
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        random_state: Optional[int] = 42,
    ):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)
    
    def resample(self, data: np.ndarray) -> np.ndarray:
        """Generate maximum entropy bootstrap sample.
        
        Simplified implementation based on:
        1. Sort data to get order statistics
        2. Generate uniform random quantiles
        3. Interpolate between order statistics
        4. Preserve original ordering via ranking
        """
        n = len(data)
        
        # Get ranks and sorted data
        ranks = stats.rankdata(data, method='ordinal')
        sorted_data = np.sort(data)
        
        # Generate noise for perturbation
        # Use order statistics quantiles with small noise
        quantiles = (ranks - 0.5) / n
        noise = self.rng.uniform(-0.5/n, 0.5/n, size=n)
        perturbed_q = np.clip(quantiles + noise, 0.001, 0.999)
        
        # Interpolate to get bootstrap values
        boot_values = np.interp(perturbed_q, np.linspace(0, 1, n), sorted_data)
        
        return boot_values
    
    def bootstrap_statistic(
        self,
        data: np.ndarray,
        statistic_func: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """Compute bootstrap distribution of a statistic."""
        stats_array = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            boot_data = self.resample(data)
            stats_array[i] = statistic_func(boot_data)
        
        return stats_array


def compute_sharpe(returns: np.ndarray, annual_factor: float = 252) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(annual_factor)


class StrategyStressTest:
    """Complete stress testing pipeline for trading strategies.
    
    Combines:
    - Bootstrap confidence intervals
    - Deflated Sharpe Ratio
    - Statistical significance testing
    
    Example:
        >>> tester = StrategyStressTest(n_simulations=1000)
        >>> result = tester.run(strategy_returns)
        >>> print(f"Deflated Sharpe: {result.deflated_sharpe:.2f}")
        >>> print(f"P-value: {result.pvalue:.4f}")
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        block_size: int = 20,
        confidence_level: float = 0.95,
        n_trials: int = 1,
        bootstrap_method: str = "block",  # "block" or "meboot"
        random_state: int = 42,
    ):
        """Initialize stress tester.
        
        Args:
            n_simulations: Number of bootstrap samples.
            block_size: Block size for block bootstrap.
            confidence_level: Confidence level for significance tests.
            n_trials: Number of strategy variants tested (for multiple testing).
            bootstrap_method: "block" or "meboot" (max entropy).
            random_state: Random seed.
        """
        self.n_simulations = n_simulations
        self.block_size = block_size
        self.confidence_level = confidence_level
        self.n_trials = n_trials
        self.bootstrap_method = bootstrap_method
        self.random_state = random_state
        
        # Initialize bootstrap sampler
        if bootstrap_method == "meboot":
            self.bootstrap = MaxEntropyBootstrap(n_simulations, random_state)
        else:
            self.bootstrap = BlockBootstrap(block_size, n_simulations, random_state)
    
    def run(self, returns: np.ndarray) -> StressTestResult:
        """Run complete stress test.
        
        Args:
            returns: Array of strategy returns.
            
        Returns:
            StressTestResult with all statistics.
        """
        # 1. Observed Sharpe
        observed_sharpe = compute_sharpe(returns)
        
        # 2. Bootstrap Sharpe distribution
        bootstrap_sharpes = self.bootstrap.bootstrap_statistic(returns, compute_sharpe)
        
        mean_boot = np.mean(bootstrap_sharpes)
        std_boot = np.std(bootstrap_sharpes)
        
        # 3. Deflated Sharpe
        _, deflated_sharpe, pvalue = DeflatedSharpeRatio.calculate(
            returns, 
            n_trials=self.n_trials
        )
        
        # 4. Significance test (one-sided: is SR > 0?)
        # Use bootstrap percentile
        alpha = 1 - self.confidence_level
        lower_bound = np.percentile(bootstrap_sharpes, alpha * 100)
        is_significant = lower_bound > 0 and pvalue < alpha
        
        return StressTestResult(
            observed_sharpe=observed_sharpe,
            mean_bootstrap_sharpe=mean_boot,
            std_bootstrap_sharpe=std_boot,
            pvalue=pvalue,
            deflated_sharpe=deflated_sharpe,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            n_simulations=self.n_simulations,
            bootstrap_sharpes=bootstrap_sharpes,
        )
    
    def run_from_nav(self, nav_series: np.ndarray) -> StressTestResult:
        """Run stress test from NAV series.
        
        Args:
            nav_series: Array of portfolio NAV values.
            
        Returns:
            StressTestResult.
        """
        returns = np.diff(nav_series) / nav_series[:-1]
        return self.run(returns)


def stress_test_strategy(
    returns: pd.Series,
    n_simulations: int = 1000,
    n_trials: int = 1,
    confidence: float = 0.95,
) -> StressTestResult:
    """Convenience function for strategy stress testing.
    
    Args:
        returns: Series of strategy returns.
        n_simulations: Bootstrap samples.
        n_trials: Number of strategies tested.
        confidence: Confidence level.
        
    Returns:
        StressTestResult.
    """
    tester = StrategyStressTest(
        n_simulations=n_simulations,
        n_trials=n_trials,
        confidence_level=confidence,
    )
    
    return tester.run(returns.values)
