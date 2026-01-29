"""
Regime Detection Module (Track 3)
=================================

Uses Hidden Markov Models (HMM) to classify market regimes:
- Bull: Low volatility, positive drift
- Bear: High volatility, negative drift  
- Sideways: Medium volatility, near-zero drift

The detected regime is used to:
1. Select the appropriate causal graph topology
2. Adjust strategy parameters (leverage, position sizing)
3. Provide context in the dashboard sidebar

References:
    - Hamilton (1989) "A New Approach to the Economic Analysis of 
      Nonstationary Time Series and the Business Cycle"
    - Mulvey & Bilgili (2018) "Regime-Switching Models for Dynamic
      Asset Allocation"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "Bull"
    BEAR = "Bear"
    SIDEWAYS = "Sideways"


@dataclass
class RegimeResult:
    """Result of regime detection."""
    current_regime: MarketRegime
    regime_history: pd.Series
    regime_probabilities: pd.DataFrame
    transition_matrix: np.ndarray
    regime_means: np.ndarray
    regime_volatilities: np.ndarray


class HMMRegimeDetector:
    """Hidden Markov Model for market regime detection.
    
    The model uses two observable features:
    1. Rolling returns (momentum signal)
    2. Rolling volatility (risk signal)
    
    These are modeled as emissions from 3 hidden states (regimes).
    
    Example:
        >>> detector = HMMRegimeDetector(n_regimes=3, lookback=20)
        >>> result = detector.fit_predict(returns)
        >>> print(f"Current regime: {result.current_regime}")
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback: int = 20,
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """Initialize HMM regime detector.
        
        Args:
            n_regimes: Number of hidden states (default 3: Bull/Bear/Sideways).
            lookback: Rolling window for feature calculation.
            n_iter: Number of EM iterations for HMM fitting.
            random_state: Random seed for reproducibility.
        """
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self._regime_mapping: dict = {}
        
    def _compute_features(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute observable features for HMM.
        
        Features:
        1. Rolling mean return (momentum)
        2. Rolling volatility (risk)
        
        Args:
            returns: DataFrame of asset returns (index: Date, columns: Tickers).
            
        Returns:
            Array of shape (n_obs, 2) with [return, volatility] features.
        """
        # Use market return (mean across assets)
        market_return = returns.mean(axis=1)
        
        # Rolling features
        rolling_return = market_return.rolling(self.lookback).mean()
        rolling_vol = market_return.rolling(self.lookback).std()
        
        # Stack and drop NaN
        features = pd.DataFrame({
            'return': rolling_return,
            'volatility': rolling_vol
        }).dropna()
        
        return features.values, features.index
    
    def _assign_regime_labels(self, means: np.ndarray) -> dict:
        """Map hidden states to interpretable regime names.
        
        Logic:
        - State with highest mean return -> Bull
        - State with lowest mean return -> Bear
        - Middle state -> Sideways
        
        Args:
            means: Array of shape (n_regimes, 2) with [return, vol] means.
            
        Returns:
            Dictionary mapping state index to MarketRegime.
        """
        # Sort by mean return
        return_order = np.argsort(means[:, 0])
        
        mapping = {}
        mapping[return_order[0]] = MarketRegime.BEAR      # Lowest return
        mapping[return_order[-1]] = MarketRegime.BULL     # Highest return
        
        # Middle state(s) are Sideways
        for i in return_order[1:-1]:
            mapping[i] = MarketRegime.SIDEWAYS
            
        # Handle edge case of only 2 regimes
        if self.n_regimes == 2:
            mapping[return_order[0]] = MarketRegime.BEAR
            mapping[return_order[1]] = MarketRegime.BULL
            
        return mapping
    
    def fit_predict(self, returns: pd.DataFrame) -> RegimeResult:
        """Fit HMM and predict regimes for the entire history.
        
        Args:
            returns: DataFrame of asset returns.
            
        Returns:
            RegimeResult with current regime, history, and model parameters.
        """
        if not HMM_AVAILABLE:
            # Fallback to simple volatility-based regime
            return self._fallback_regime_detection(returns)
        
        # Compute features
        features, dates = self._compute_features(returns)
        
        if len(features) < self.lookback * 2:
            raise ValueError(f"Insufficient data: need at least {self.lookback * 2} observations")
        
        # Fit HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(features)
        
        # Predict hidden states
        hidden_states = self.model.predict(features)
        state_probs = self.model.predict_proba(features)
        
        # Map states to interpretable regimes
        self._regime_mapping = self._assign_regime_labels(self.model.means_)
        
        # Convert to regime labels
        regime_labels = [self._regime_mapping[s].value for s in hidden_states]
        regime_history = pd.Series(regime_labels, index=dates, name='Regime')
        
        # Regime probabilities DataFrame
        prob_cols = [self._regime_mapping[i].value for i in range(self.n_regimes)]
        regime_probs = pd.DataFrame(
            state_probs, 
            index=dates, 
            columns=[f"P({r})" for r in prob_cols]
        )
        
        # Extract regime characteristics
        regime_vols = np.sqrt(np.diagonal(self.model.covars_, axis1=1, axis2=2)[:, 1])
        
        return RegimeResult(
            current_regime=self._regime_mapping[hidden_states[-1]],
            regime_history=regime_history,
            regime_probabilities=regime_probs,
            transition_matrix=self.model.transmat_,
            regime_means=self.model.means_[:, 0],  # Return means
            regime_volatilities=regime_vols,
        )
    
    def _fallback_regime_detection(self, returns: pd.DataFrame) -> RegimeResult:
        """Simple volatility-based regime detection when hmmlearn unavailable.
        
        Uses quantiles of rolling volatility to assign regimes.
        """
        market_return = returns.mean(axis=1)
        rolling_vol = market_return.rolling(self.lookback).std().dropna()
        rolling_ret = market_return.rolling(self.lookback).mean().dropna()
        
        # Align
        dates = rolling_vol.index.intersection(rolling_ret.index)
        rolling_vol = rolling_vol.loc[dates]
        rolling_ret = rolling_ret.loc[dates]
        
        # Quantile thresholds
        vol_low = rolling_vol.quantile(0.33)
        vol_high = rolling_vol.quantile(0.67)
        
        # Assign regimes
        regimes = []
        for date in dates:
            vol = rolling_vol.loc[date]
            ret = rolling_ret.loc[date]
            
            if ret > 0 and vol < vol_high:
                regimes.append(MarketRegime.BULL.value)
            elif ret < 0 and vol > vol_low:
                regimes.append(MarketRegime.BEAR.value)
            else:
                regimes.append(MarketRegime.SIDEWAYS.value)
        
        regime_history = pd.Series(regimes, index=dates, name='Regime')
        
        # Mock probabilities
        regime_probs = pd.DataFrame({
            'P(Bull)': (regime_history == 'Bull').astype(float),
            'P(Bear)': (regime_history == 'Bear').astype(float),
            'P(Sideways)': (regime_history == 'Sideways').astype(float),
        }, index=dates)
        
        current = MarketRegime(regimes[-1]) if regimes else MarketRegime.SIDEWAYS
        
        return RegimeResult(
            current_regime=current,
            regime_history=regime_history,
            regime_probabilities=regime_probs,
            transition_matrix=np.eye(3) / 3,  # Placeholder
            regime_means=np.array([0.001, -0.001, 0.0]),
            regime_volatilities=np.array([0.01, 0.02, 0.015]),
        )
    
    def predict_current(self, returns: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Predict regime for the most recent observation.
        
        Args:
            returns: DataFrame of asset returns.
            
        Returns:
            Tuple of (current regime, confidence probability).
        """
        result = self.fit_predict(returns)
        
        # Get probability of current regime
        prob_col = f"P({result.current_regime.value})"
        confidence = result.regime_probabilities[prob_col].iloc[-1]
        
        return result.current_regime, confidence


def get_regime_style(regime: MarketRegime) -> dict:
    """Get display styling for a regime (Flexoki colors).
    
    Returns:
        Dict with 'color', 'icon', and 'strategy' keys.
    """
    styles = {
        MarketRegime.BULL: {
            'color': '#879A39',  # Flexoki Green
            'icon': '🐂',
            'strategy': 'Mean Reversion / Leverage Up',
            'delta_color': 'normal',
        },
        MarketRegime.BEAR: {
            'color': '#D14D41',  # Flexoki Red
            'icon': '🐻',
            'strategy': 'Defensive / Short Bias',
            'delta_color': 'inverse',
        },
        MarketRegime.SIDEWAYS: {
            'color': '#205EA6',  # Flexoki Blue
            'icon': '🌊',
            'strategy': 'Neutral / Range Trading',
            'delta_color': 'off',
        },
    }
    return styles.get(regime, styles[MarketRegime.SIDEWAYS])
