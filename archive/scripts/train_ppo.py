"""
Train PPO Agent for Multi-Asset Trading
========================================

Uses the RL environment with pretrained GNN embeddings.

Usage:
    python scripts/train_ppo.py --timesteps 50000 --reward log_return
"""

import argparse
from pathlib import Path
import pandas as pd

from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.config import default_config
from cointegration_gnn.rl.ppo_agent import TradingAgent, train_ppo_agent


def main():
    parser = argparse.ArgumentParser(description="Train PPO Trading Agent")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training steps")
    parser.add_argument("--reward", choices=["log_return", "diff_sharpe"], default="log_return")
    parser.add_argument("--save-path", default="models/ppo_trader.zip")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PPO Trading Agent Training")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/4] Loading market data...")
    loader = DataLoader(tickers=default_config.data.tickers)
    prices, returns = loader.fetch_data(
        start=default_config.data.train_start,
        end=default_config.data.val_end
    )
    features = loader.compute_features(prices, returns)
    
    print(f"  → Prices: {prices.shape}")
    print(f"  → Features: {features.shape}")
    
    # 2. Create Environment
    print(f"\n[2/4] Creating trading environment (reward: {args.reward})...")
    agent = TradingAgent.from_data(
        prices, features,
        reward_type=args.reward,
        tensorboard_log="logs/ppo_tensorboard",
        verbose=1,
    )
    
    print(f"  → Observation space: {agent.env.observation_space.shape}")
    print(f"  → Action space: {agent.env.action_space.shape}")
    
    # 3. Train
    print(f"\n[3/4] Training PPO for {args.timesteps:,} steps...")
    agent.train(
        total_timesteps=args.timesteps,
        checkpoint_freq=10_000,
        checkpoint_dir="models/rl_checkpoints",
    )
    
    # 4. Evaluate
    print("\n[4/4] Evaluating agent...")
    stats = agent.evaluate(n_episodes=10)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Mean Return:     {stats['mean_return']:.2%} (±{stats['std_return']:.2%})")
    print(f"  Mean Sharpe:     {stats['mean_sharpe']:.2f} (±{stats['std_sharpe']:.2f})")
    print(f"  Mean Max DD:     {stats['mean_max_dd']:.2%}")
    print(f"  Mean Final NAV:  {stats['mean_nav']:.4f}")
    
    # 5. Save
    agent.save(args.save_path)
    print(f"\n✅ Model saved to {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
