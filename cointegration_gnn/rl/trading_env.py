"""
Multi-Asset Trading Environment (Track 1: MARL)
================================================

Gymnasium-compatible environment for training RL agents on portfolio allocation.
Supports multiple reward functions:
- Log Returns (simple, stable training)
- Differential Sharpe Ratio (risk-adjusted, advanced)

Reference:
    - Moody & Saffell (2001) "Learning to Trade via Direct Reinforcement"
    - Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"

Example:
    >>> env = TradingEnv(prices, features, reward_type="log_return")
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, done, truncated, info = env.step(action)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    """Configuration for the trading environment."""
    transaction_cost_bps: float = 10.0  # 10 bps per trade
    max_position: float = 0.25  # Max position per asset (25%)
    reward_type: str = "log_return"  # "log_return" or "diff_sharpe"
    lookback_window: int = 20  # Observation window size
    episode_length: Optional[int] = None  # None = use all data


class TradingEnv(gym.Env):
    """Multi-asset portfolio allocation environment.
    
    Observation Space:
        - Asset features (returns, volatility, momentum, etc.)
        - Current positions
        - Portfolio metrics (cash, NAV)
        
    Action Space:
        - Continuous: Target weights for each asset [-max_pos, +max_pos]
        
    Reward:
        - Log Returns: Simple log(NAV_t / NAV_{t-1})
        - Differential Sharpe: Risk-adjusted reward with running estimates
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize trading environment.
        
        Args:
            prices: DataFrame with price data (index: Date, columns: Tickers).
            features: DataFrame with feature data (MultiIndex: Date x Ticker x Features).
            config: Environment configuration.
            render_mode: Rendering mode ("human" or None).
        """
        super().__init__()
        
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        
        # Store data
        self.prices = prices
        self.features = features
        self.tickers = list(prices.columns)
        self.n_assets = len(self.tickers)
        
        # Align dates
        self.dates = prices.index.intersection(features.index.get_level_values(0).unique())
        self.dates = sorted(self.dates)[self.config.lookback_window:]
        
        # Compute returns
        self.returns = prices.pct_change().fillna(0)
        
        # Determine feature dimension
        sample_date = self.dates[0]
        sample_features = features.loc[sample_date]
        if hasattr(sample_features, 'values'):
            self.n_features = sample_features.shape[-1] if len(sample_features.shape) > 1 else 1
        else:
            self.n_features = 8  # Default
        
        # State variables
        self.current_step = 0
        self.positions = np.zeros(self.n_assets)
        self.nav = 1.0  # Normalized NAV
        self.nav_history = [1.0]
        
        # For Differential Sharpe
        self.A = 0.0  # Running mean of returns
        self.B = 0.0  # Running mean of squared returns
        self.eta = 0.01  # Decay rate
        
        # Define spaces
        # Observation: [asset_features (n_assets * n_features) + positions (n_assets) + nav (1)]
        obs_dim = self.n_assets * self.n_features + self.n_assets + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action: target weights for each asset
        self.action_space = spaces.Box(
            low=-self.config.max_position,
            high=self.config.max_position,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        date = self.dates[self.current_step]
        
        # Get features for this date
        try:
            date_features = self.features.loc[date]
            if hasattr(date_features, 'values'):
                feat_array = date_features.values.flatten()
            else:
                feat_array = np.zeros(self.n_assets * self.n_features)
        except KeyError:
            feat_array = np.zeros(self.n_assets * self.n_features)
        
        # Pad or truncate to expected size
        expected_size = self.n_assets * self.n_features
        if len(feat_array) < expected_size:
            feat_array = np.pad(feat_array, (0, expected_size - len(feat_array)))
        else:
            feat_array = feat_array[:expected_size]
        
        # Combine: features + positions + NAV
        obs = np.concatenate([
            feat_array,
            self.positions,
            [self.nav]
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, prev_nav: float, curr_nav: float, prev_positions: np.ndarray) -> float:
        """Compute reward based on selected reward type."""
        if self.config.reward_type == "log_return":
            # Simple log return
            if prev_nav > 0 and curr_nav > 0:
                return np.log(curr_nav / prev_nav)
            return 0.0
            
        elif self.config.reward_type == "diff_sharpe":
            # Differential Sharpe Ratio (Moody & Saffell, 2001)
            # DSR = (B * ΔA - 0.5 * A * ΔB) / (B - A^2)^1.5
            
            R_t = (curr_nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
            
            delta_A = R_t - self.A
            delta_B = R_t**2 - self.B
            
            # Update running estimates
            self.A = self.A + self.eta * delta_A
            self.B = self.B + self.eta * delta_B
            
            # Compute DSR
            denom = self.B - self.A**2
            if denom > 1e-8:
                dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / (denom ** 1.5)
            else:
                dsr = 0.0
            
            return dsr
        
        else:
            return 0.0
    
    def _apply_transaction_costs(self, old_positions: np.ndarray, new_positions: np.ndarray) -> float:
        """Calculate transaction costs from position changes."""
        turnover = np.abs(new_positions - old_positions).sum()
        cost = turnover * (self.config.transaction_cost_bps / 10000)
        return cost
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.positions = np.zeros(self.n_assets)
        self.nav = 1.0
        self.nav_history = [1.0]
        self.A = 0.0
        self.B = 0.0
        
        obs = self._get_observation()
        info = {"date": self.dates[0], "nav": self.nav}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.
        
        Args:
            action: Target portfolio weights.
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action to valid range
        target_positions = np.clip(action, -self.config.max_position, self.config.max_position)
        
        # Store previous state
        prev_nav = self.nav
        prev_positions = self.positions.copy()
        
        # Get current date and returns
        date = self.dates[self.current_step]
        date_returns = self.returns.loc[date].values
        
        # Apply transaction costs
        tx_cost = self._apply_transaction_costs(prev_positions, target_positions)
        
        # Update positions
        self.positions = target_positions
        
        # Portfolio return (position-weighted returns minus costs)
        portfolio_return = np.dot(self.positions, date_returns) - tx_cost
        
        # Update NAV
        self.nav = self.nav * (1 + portfolio_return)
        self.nav_history.append(self.nav)
        
        # Compute reward
        reward = self._compute_reward(prev_nav, self.nav, prev_positions)
        
        # Advance step
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= len(self.dates) - 1
        truncated = False
        
        if self.config.episode_length and self.current_step >= self.config.episode_length:
            truncated = True
        
        # Bankruptcy check
        if self.nav <= 0.0:
            terminated = True
            reward = -10.0  # Large penalty
        
        # Get new observation
        obs = self._get_observation()
        
        info = {
            "date": date,
            "nav": self.nav,
            "positions": self.positions.copy(),
            "portfolio_return": portfolio_return,
            "transaction_cost": tx_cost,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (text mode)."""
        if self.render_mode == "human":
            date = self.dates[self.current_step]
            print(f"Step {self.current_step} | Date: {date} | NAV: {self.nav:.4f}")
            print(f"  Positions: {dict(zip(self.tickers, self.positions))}")
    
    def get_episode_stats(self) -> Dict:
        """Calculate episode statistics."""
        nav_array = np.array(self.nav_history)
        returns = np.diff(nav_array) / nav_array[:-1]
        
        total_return = (nav_array[-1] / nav_array[0]) - 1
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        peak = np.maximum.accumulate(nav_array)
        drawdown = (peak - nav_array) / peak
        max_dd = drawdown.max()
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "final_nav": nav_array[-1],
            "n_steps": len(nav_array) - 1,
        }


def make_trading_env(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    reward_type: str = "log_return",
    **kwargs
) -> TradingEnv:
    """Factory function to create trading environment.
    
    Args:
        prices: Price data DataFrame.
        features: Feature data DataFrame.
        reward_type: "log_return" or "diff_sharpe".
        **kwargs: Additional config parameters.
        
    Returns:
        Configured TradingEnv instance.
    """
    config = EnvConfig(reward_type=reward_type, **kwargs)
    return TradingEnv(prices, features, config)
