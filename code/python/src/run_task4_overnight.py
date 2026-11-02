#!/usr/bin/env python3
"""
Task 4: Full Spectral Motif Extraction & Comparison vs. Jonathan Mishler's Reference Templates.

Usage:
    python run_task4_overnight.py --data_path /path/to/lfp_file.mat --ref_path /path/to/finalGlutLFPMotifs.mat
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from src.smooth_env import smooth_env
from src.get_motifs import get_motifs


# 1. Logging setup
def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "task4_overnight.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging initialized. Output dir: {output_dir}")


# 2. Canonical 6 bands from Mishler et al.
MISHLER_6_BANDS = np.array([
    [1.0, 4.0],      # Delta
    [4.0, 8.0],      # Theta
    [8.0, 12.0],     # Alpha
    [12.0, 30.0],    # Beta
    [30.0, 60.0],    # Low-Gamma
    [60.0, 100.0]    # High-Gamma
], dtype=np.float64)

BAND_NAMES = ["Delta", "Theta", "Alpha", "Beta", "Low-Gamma", "High-Gamma"]


def parse_args():
    parser = argparse.ArgumentParser(description="Task 4 Overnight Full Motif Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to MAT file containing lfp variable")
    parser.add_argument("--ref_path", type=str, required=True, help="Path to finalGlutLFPMotifs.mat")
    parser.add_argument("--output_dir", type=str, default="./results_task4", help="Directory to save results")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling frequency (Hz)")
    parser.add_argument("--n_null", type=int, default=20, help="Number of circular shift null iterations")
    parser.add_argument("--state_col", type=int, default=0, help="Column index for brain state (0 = Awake)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for null shifts")
    return parser.parse_args()


def load_data(data_path: str, state_col: int):
    logging.info(f"Loading LFP matrix from: {data_path}")
    mat = sio.loadmat(data_path)
    
    if "LFP" not in mat:
        raise KeyError(f"'LFP' variable not found in {data_path}. Available keys: {list(mat.keys())}")
    
    lfp_mat = mat["LFP"]
    n_rows, n_cols = lfp_mat.shape
    logging.info(f"Loaded 'LFP' matrix with shape: ({n_rows}, {n_cols}). Selecting state col {state_col}.")
    
    sessions_raw = []
    for r in range(n_rows):
        entry = lfp_mat[r, state_col]
        while isinstance(entry, np.ndarray) and entry.dtype == object:
            entry = entry[0]
        raw_1d = np.squeeze(entry).astype(np.float64)
        sessions_raw.append(raw_1d)
        logging.info(f"  Session {r}: {raw_1d.size:,} samples ({raw_1d.size / 250.0 / 60.0:.2f} mins)")
        
    return sessions_raw


def run_preprocessing(sessions_raw, fs: float):
    logging.info("Starting envelope extraction and z-scoring across sessions...")
    envelopes_list = []
    
    for idx, trace in enumerate(sessions_raw):
        t0 = time.time()
        # Compute smooth Hilbert envelopes
        env = smooth_env(trace, fs=fs, freq_bands=MISHLER_6_BANDS)
        
        # Exclude initial/terminal NaNs
        valid = np.isfinite(env).all(axis=1)
        env_valid = env[valid]
        
        # get_motifs expects (n_bands, n_samples)
        envelopes_list.append(env_valid.T)
        elapsed = time.time() - t0
        logging.info(f"  Processed Session {idx + 1}/{len(sessions_raw)}: shape {env_valid.T.shape} in {elapsed:.1f}s")
        
    return envelopes_list


def compare_with_reference(discovered_motifs: np.ndarray, final_motifs_ref: np.ndarray):
    """
    Evaluates discovered motif vectors against Jonathan's reference matrix (76 x 6).
    Accounts for positive poles, opponent negative poles, and global sign invariance.
    """
    n_bands, n_poles = discovered_motifs.shape
    n_pairs = n_poles // 2
    n_ref = final_motifs_ref.shape[0]
    
    logging.info(f"Comparing {n_pairs} discovered motif pairs ({n_poles} total poles) against {n_ref} reference rows...")
    
    results = []
    
    for p_idx in range(n_poles):
        pole_vec = discovered_motifs[:, p_idx]
        pole_label = f"M{(p_idx % n_pairs) + 1}{'+' if p_idx < n_pairs else '-'}"
        
        corrs = np.zeros(n_ref)
        cos_sims = np.zeros(n_ref)
        
        for r_idx in range(n_ref):
            ref_vec = final_motifs_ref[r_idx, :]
            
            # Pearson r
            r = np.corrcoef(pole_vec, ref_vec)[0, 1]
            corrs[r_idx] = r
            
            # Cosine similarity
            norm_prod = np.linalg.norm(pole_vec) * np.linalg.norm(ref_vec)
            cos_sims[r_idx] = np.dot(pole_vec, ref_vec) / norm_prod if norm_prod > 0 else 0.0
            
        best_r_idx = int(np.nanargmax(corrs))
        best_r = corrs[best_r_idx]
        
        worst_r_idx = int(np.nanargmin(corrs))
        worst_r = corrs[worst_r_idx]
        
        # Determine whether direct or inverted alignment is stronger
        if abs(best_r) >= abs(worst_r):
            match_row = best_r_idx
            aligned_r = best_r
            alignment = "Direct"
            matched_cos = cos_sims[best_r_idx]
        else:
            match_row = worst_r_idx
            aligned_r = worst_r
            alignment = "Inverted (Opponent)"
            matched_cos = cos_sims[worst_r_idx]
            
        results.append({
            "Discovered Pole": pole_label,
            "Pole Index": p_idx,
            "Best Ref Row": match_row,
            "Alignment": alignment,
            "Pearson r": aligned_r,
            "Absolute r": abs(aligned_r),
            "Cosine Similarity": matched_cos,
            "Status": "HIGH CONCORDANCE" if abs(aligned_r) >= 0.85 else ("MODERATE" if abs(aligned_r) >= 0.70 else "LOW")
        })
        
    return pd.DataFrame(results)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    
    start_total_time = time.time()
    logging.info("Starting Task 4 Overnight Execution Pipeline...")
    
    # 1. Load Reference finalMotifs
    logging.info(f"Loading reference motifs: {args.ref_path}")
    ref_data = sio.loadmat(args.ref_path)
    if "finalMotifs" not in ref_data:
        raise KeyError(f"'finalMotifs' not found in {args.ref_path}")
    final_motifs_ref = ref_data["finalMotifs"]
    logging.info(f"Loaded 'finalMotifs' with shape: {final_motifs_ref.shape}")
    
    # 2. Load Raw LFP
    sessions_raw = load_data(args.data_path, args.state_col)
    
    # 3. Preprocess Envelopes
    all_envelopes = run_preprocessing(sessions_raw, args.fs)
    
    # 4. Extract Motifs with Null Significance Testing
    logging.info(f"Launching get_motifs with {args.n_null} null iterations (seed={args.seed})...")
    t0_discovery = time.time()
    
    result = get_motifs(
        all_envelopes,
        fs=args.fs,
        n_null=args.n_null,
        seed=args.seed,
        verbose=True
    )
    
    discovery_duration = time.time() - t0_discovery
    logging.info(f"Motif extraction finished in {discovery_duration / 60.0:.2f} minutes.")
    logging.info(f"Final Discovered Motif Pairs: {result.n_motifs}")
    
    # 5. Evaluate Parity & Save Results
    if result.n_motifs == 0:
        logging.warning("No motif pairs passed the null hypothesis significance threshold!")
        return
        
    discovered_motifs = result.motifs  # Shape: (6, 2 * n_motifs)
    logging.info(f"Discovered motif matrix shape: {discovered_motifs.shape}")
    
    # Compare with reference
    comparison_df = compare_with_reference(discovered_motifs, final_motifs_ref)
    
    # Display table in log
    logging.info("\n=== TASK 4 COMPARISON TABLE VS. JONATHAN'S REFERENCE ===")
    logging.info("\n" + comparison_df.to_string(index=False))
    
    # Save artifacts
    csv_path = output_dir / "task4_motif_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    
    np.save(output_dir / "discovered_motifs.npy", discovered_motifs)
    sio.savemat(output_dir / "task4_output.mat", {
        "motifs": discovered_motifs,
        "n_motifs": result.n_motifs,
        "comparison_table": comparison_df.to_dict(orient="records")
    })
    
    total_elapsed = time.time() - start_total_time
    logging.info(f"Task 4 completed successfully in {total_elapsed / 60.0:.2f} minutes.")
    logging.info(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()