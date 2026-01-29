
import subprocess
import pandas as pd
import itertools
from pathlib import Path
import sys

def run_tuning():
    print("🎛️  Starting Hyperparameter Tuning...")
    
    # Define Parameter Grid
    # Focused on Turnover Penalty as requested
    param_grid = {
        'turnover_penalty': [0.0, 0.05, 0.1, 0.5],
        'learning_rate': [1e-3] 
    }
    
    # Create experiment name for isolation
    experiment_name = "tuning_experiment"
    
    # Generate combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    total_runs = len(combinations)
    
    for i, params in enumerate(combinations):
        print(f"\n[Run {i+1}/{total_runs}] Testing params: {params}")
        
        # Construct command
        cmd = [
            sys.executable, "scripts/train.py",
            "--turnover_penalty", str(params['turnover_penalty']),
            "--learning_rate", str(params['learning_rate']),
            "--experiment_name", experiment_name,
            "--max_epochs", "15" # Reduced epochs for faster tuning
        ]
        
        try:
            # Run training
            subprocess.run(cmd, check=True)
            
            # Retrieve results from logs
            # Find the latest version directory in logs/tuning_experiment/
            log_dir = Path("logs") / experiment_name
            
            # Filter directories starting with 'version_' and sort by number
            versions = sorted(
                [d for d in log_dir.glob("version_*") if d.is_dir()],
                key=lambda x: int(x.name.split('_')[1])
            )
            
            if not versions:
                print("⚠️  No log directory found for this run.")
                continue
                
            latest_version = versions[-1]
            metrics_path = latest_version / "metrics.csv"
            
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                
                # Check if we have valid data
                if 'val/IC' in df.columns and 'val/return' in df.columns:
                    # Get the metrics from the Best Epoch (based on max val/IC)
                    best_epoch_idx = df['val/IC'].idxmax()
                    best_row = df.loc[best_epoch_idx]
                    
                    res = params.copy()
                    res['val_IC_max'] = best_row['val/IC']
                    res['val_return_at_max_IC'] = best_row['val/return']
                    res['train_loss'] = best_row.get('train/loss', 0)
                    res['version'] = latest_version.name
                    results.append(res)
                    
                    print(f"   ✅ Result: IC={res['val_IC_max']:.4f}, Return={res['val_return_at_max_IC']:.4f}")
                else:
                    print("⚠️  Metrics columns missing in CSV.")
            else:
                print(f"⚠️  metrics.csv not found in {latest_version}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed for {params}")
            continue
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            continue
            
    # Save Final Results
    if results:
        results_df = pd.read_csv("tuning_results.csv") if Path("tuning_results.csv").exists() else pd.DataFrame()
        new_results = pd.DataFrame(results)
        
        # Combine if needed, or just overwrite for this session
        final_df = pd.concat([results_df, new_results], ignore_index=True)
        final_df.to_csv("tuning_results.csv", index=False)
        
        print("\n🏆 Tuning Complete! Top 3 Configurations:")
        print(final_df.sort_values(by='val_IC_max', ascending=False).head(3))
        print(f"Results saved to {Path('.').resolve() / 'tuning_results.csv'}")
    else:
        print("\n❌ No results collected.")

if __name__ == "__main__":
    run_tuning()
