"""
PPO Agent for Multi-Asset Trading (Track 1: MARL)
==================================================

Wrapper around stable-baselines3 PPO for the trading environment.
Includes:
- Custom policy network with GNN embeddings (optional)
- Training with checkpointing
- Evaluation utilities

Example:
    >>> agent = TradingAgent(env, policy="MlpPolicy")
    >>> agent.train(total_timesteps=100_000)
    >>> agent.save("models/ppo_trader.zip")
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, 
        EvalCallback,
        CallbackList
    )
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from cointegration_gnn.rl.trading_env import TradingEnv, make_trading_env


class TradingAgent:
    """PPO-based trading agent.
    
    Wraps stable-baselines3 PPO with sensible defaults for financial applications.
    
    Attributes:
        env: The trading environment.
        model: The PPO model instance.
        
    Example:
        >>> agent = TradingAgent.from_data(prices, features)
        >>> agent.train(total_timesteps=50_000)
        >>> stats = agent.evaluate(n_episodes=10)
    """
    
    def __init__(
        self,
        env: TradingEnv,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        device: str = "auto",
    ):
        """Initialize PPO agent.
        
        Args:
            env: Trading environment instance.
            policy: Policy architecture ("MlpPolicy" or custom).
            learning_rate: Adam optimizer learning rate.
            n_steps: Steps to run per update.
            batch_size: Minibatch size for updates.
            n_epochs: Epochs per update.
            gamma: Discount factor.
            gae_lambda: GAE lambda for advantage estimation.
            clip_range: PPO clipping range.
            ent_coef: Entropy coefficient for exploration.
            verbose: Verbosity level.
            tensorboard_log: Path for TensorBoard logs.
            device: "cpu", "cuda", or "auto".
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3"
            )
        
        self.env = env
        
        # Wrap in DummyVecEnv for SB3 compatibility
        self._vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        # Initialize PPO
        self.model = PPO(
            policy=policy,
            env=self._vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            device=device,
        )
        
    @classmethod
    def from_data(
        cls,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        reward_type: str = "log_return",
        **kwargs
    ) -> "TradingAgent":
        """Create agent from price/feature data.
        
        Args:
            prices: Price DataFrame.
            features: Feature DataFrame.
            reward_type: "log_return" or "diff_sharpe".
            **kwargs: Additional arguments for TradingAgent.__init__.
            
        Returns:
            Configured TradingAgent instance.
        """
        env = make_trading_env(prices, features, reward_type=reward_type)
        return cls(env, **kwargs)
    
    def train(
        self,
        total_timesteps: int = 100_000,
        checkpoint_freq: int = 10_000,
        checkpoint_dir: str = "models/rl_checkpoints",
        eval_freq: Optional[int] = None,
        eval_env: Optional[TradingEnv] = None,
    ) -> "TradingAgent":
        """Train the PPO agent.
        
        Args:
            total_timesteps: Total environment steps for training.
            checkpoint_freq: Save checkpoint every N steps.
            checkpoint_dir: Directory for checkpoints.
            eval_freq: Evaluate every N steps (optional).
            eval_env: Separate environment for evaluation.
            
        Returns:
            Self for chaining.
        """
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix="ppo_trader",
            verbose=1,
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback (optional)
        if eval_freq and eval_env:
            eval_vec_env = DummyVecEnv([lambda: Monitor(eval_env)])
            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=checkpoint_dir,
                log_path=checkpoint_dir,
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
        
        return self
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given observation.
        
        Args:
            obs: Observation array.
            deterministic: If True, use deterministic policy.
            
        Returns:
            Action array.
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate agent performance.
        
        Args:
            n_episodes: Number of evaluation episodes.
            deterministic: Use deterministic policy.
            
        Returns:
            Dict with mean/std of episode statistics.
        """
        all_stats = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
            
            stats = self.env.get_episode_stats()
            all_stats.append(stats)
        
        # Aggregate statistics
        df = pd.DataFrame(all_stats)
        
        return {
            "mean_return": df["total_return"].mean(),
            "std_return": df["total_return"].std(),
            "mean_sharpe": df["sharpe_ratio"].mean(),
            "std_sharpe": df["sharpe_ratio"].std(),
            "mean_max_dd": df["max_drawdown"].mean(),
            "mean_nav": df["final_nav"].mean(),
        }
    
    def save(self, path: str):
        """Save model to file."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, env: TradingEnv) -> "TradingAgent":
        """Load model from file."""
        agent = cls.__new__(cls)
        agent.env = env
        agent._vec_env = DummyVecEnv([lambda: Monitor(env)])
        agent.model = PPO.load(path, env=agent._vec_env)
        return agent


def train_ppo_agent(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    total_timesteps: int = 100_000,
    reward_type: str = "log_return",
    save_path: str = "models/ppo_trader.zip",
) -> Dict[str, float]:
    """Convenience function to train a PPO agent.
    
    Args:
        prices: Price data.
        features: Feature data.
        total_timesteps: Training steps.
        reward_type: Reward function type.
        save_path: Path to save trained model.
        
    Returns:
        Evaluation statistics.
    """
    # Create agent
    agent = TradingAgent.from_data(
        prices, features,
        reward_type=reward_type,
        tensorboard_log="logs/ppo_tensorboard",
    )
    
    # Train
    agent.train(total_timesteps=total_timesteps)
    
    # Evaluate
    stats = agent.evaluate(n_episodes=10)
    
    # Save
    agent.save(save_path)
    
    return stats
