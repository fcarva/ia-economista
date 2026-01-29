
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_training_logs(log_dir="logs/gnn_ibov_experiment"):
    """Analyze and plot training metrics from latest version."""
    path = Path(log_dir)
    versions = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("version_")], 
                      key=lambda x: int(x.name.split('_')[1]))
    
    if not versions:
        print("No logs found.")
        return
        
    latest_version = versions[-1]
    metrics_path = latest_version / "metrics.csv"
    
    if not metrics_path.exists():
        print(f"No metrics.csv found in {latest_version}")
        return
        
    print(f"Analyzing {metrics_path}...")
    df = pd.read_csv(metrics_path)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Train Loss
    if 'train/loss' in df.columns:
        train_loss = df.dropna(subset=['train/loss'])
        sns.lineplot(data=train_loss, x='step', y='train/loss', ax=axes[0, 0], label='Train Loss')
        axes[0, 0].set_title("Training Loss")
        
    # 2. Train IC vs Val IC
    if 'train/IC' in df.columns:
        train_ic = df.dropna(subset=['train/IC'])
        sns.lineplot(data=train_ic, x='step', y='train/IC', ax=axes[0, 1], label='Train IC', alpha=0.6)
        
    if 'val/IC' in df.columns:
        val_ic = df.dropna(subset=['val/IC'])
        # Map step to epoch for validation if needed, but step is fine
        sns.lineplot(data=val_ic, x='step', y='val/IC', ax=axes[0, 1], label='Val IC', linewidth=2, color='red')
        
    axes[0, 1].set_title("Information Coefficient (IC)")
    axes[0, 1].axhline(0, color='black', linestyle='--')
    
    # 3. Model Score Dispersion
    if 'train/score_std' in df.columns:
        score_std = df.dropna(subset=['train/score_std'])
        sns.lineplot(data=score_std, x='step', y='train/score_std', ax=axes[1, 0], color='green')
        axes[1, 0].set_title("Model Score Dispersion (Std)")
        
    # 4. Returns (if available)
    if 'val/return' in df.columns:
        val_ret = df.dropna(subset=['val/return'])
        sns.lineplot(data=val_ret, x='step', y='val/return', ax=axes[1, 1], color='purple')
        axes[1, 1].set_title("Validation Return Proxy")
    
    plt.tight_layout()
    output_file = f"results/training_analysis_{latest_version.name}.png"
    plt.savefig(output_file)
    print(f"Unsaved plots to {output_file}")

if __name__ == "__main__":
    analyze_training_logs()
