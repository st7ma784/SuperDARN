"""
Diagnostic script to check reward distribution in the dataset.
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rl_forecast.datamodule import RLDataModule
from rl_forecast.reward import compute_reward

def main():
    data_dir = "/data2/rl_data"
    
    print(f"Loading data from {data_dir}")
    dm = RLDataModule(
        data_dir=data_dir,
        batch_size=16,
        n_steps=3,
        gamma=0.99,
        num_input_frames=3,
        temporal_agg_frames=4,
        val_split=0.1,
    )
    dm.setup()
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    print("\n=== TRAINING DATA ===")
    all_rewards = []
    for batch_idx, batch in enumerate(train_loader):
        s, a_data, r_n, s_tn, done = batch
        
        # Also compute rewards manually to compare
        r_manual = compute_reward(s, a_data, s_tn)
        
        all_rewards.extend(r_n.cpu().numpy())
        
        print(f"Batch {batch_idx}")
        print(f"  r_n (from dataset):  mean={r_n.mean():.6e}, std={r_n.std():.6e}, min={r_n.min():.6e}, max={r_n.max():.6e}")
        print(f"  r_manual (computed): mean={r_manual.mean():.6e}, std={r_manual.std():.6e}, min={r_manual.min():.6e}, max={r_manual.max():.6e}")
        print(f"  s shape: {s.shape}, a_data shape: {a_data.shape}, s_tn shape: {s_tn.shape}")
        print(f"  s range: [{s.min():.4f}, {s.max():.4f}]")
        print(f"  a_data range: [{a_data.min():.4f}, {a_data.max():.4f}]")
        print()
        
        if batch_idx >= 4:
            break
    
    all_rewards = np.array(all_rewards)
    print(f"\n=== AGGREGATE STATS (first 5 batches) ===")
    print(f"All rewards: mean={all_rewards.mean():.6e}, std={all_rewards.std():.6e}")
    print(f"             min={all_rewards.min():.6e}, max={all_rewards.max():.6e}")
    print(f"             nonzero count: {(all_rewards != 0).sum()} / {len(all_rewards)}")
    print(f"             zero count: {(all_rewards == 0).sum()} / {len(all_rewards)}")

if __name__ == "__main__":
    main()
