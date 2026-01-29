
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_tuning():
    csv_path = Path("tuning_results.csv")
    if not csv_path.exists():
        print("Dataset not found!")
        return

    df = pd.read_csv(csv_path)
    print("Loaded tuning results:")
    print(df)
    
    # Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Bar plot: IC by Penalty (hue=LR)
    sns.barplot(data=df, x="turnover_penalty", y="val_IC_max", hue="learning_rate", palette="viridis")
    
    plt.title("Impact of Turnover Penalty on Validation IC")
    plt.ylabel("Max Validation IC")
    plt.xlabel("Turnover Penalty")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    output_path = "results/tuning_summary.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    analyze_tuning()
