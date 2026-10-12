import os
import gc
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from preprocessing import load_eeg_trace, apply_notch, smooth_env
from motif_discovery import get_motifs

def main():
    # ---------------------------------------------------------
    # 1. Configuration & Path Discovery
    # ---------------------------------------------------------
    repo_root = Path.cwd().resolve()
    while repo_root.name != "dc-allen-data-acquire" and repo_root.parent != repo_root:
        repo_root = repo_root.parent

    data_dir = repo_root / "data" / "raw"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found at: {data_dir}")

    # Set up cache and results folders
    cache_dir = repo_root / "data" / "processed" / "eeg_envelopes"
    results_dir = repo_root / "results" / "motifs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    nwb_files = sorted([f for f in data_dir.iterdir() if f.suffix.lower() == ".nwb"])
    if not nwb_files:
        nwb_files = sorted(list(data_dir.rglob("*.nwb")))
        
    if not nwb_files:
        raise FileNotFoundError(f"No .nwb files found in {data_dir}")

    print(f"Discovered {len(nwb_files)} NWB files in {data_dir.name}.")
    print(f"Envelope cache directory: {cache_dir}")
    print(f"Results directory:        {results_dir}")

    # Hardcoded target for this pipeline run
    target_electrode = 'Pz' 
    print(f"\n{'='*60}")
    print(f"STARTING MOTIF DISCOVERY FOR ELECTRODE: {target_electrode}")
    print(f"{'='*60}")

    motif_out_path = results_dir / f"final_motifs_{target_electrode}.npy"
    fig_out_path = results_dir / f"motifs_{target_electrode}.png"

    # ---------------------------------------------------------
    # 2. Checkpoint 1: Load motifs if already computed
    # ---------------------------------------------------------
    if motif_out_path.exists():
        print(f"[{target_electrode}] Found existing motif results. Skipping preprocessing...")
        final_motifs = np.load(motif_out_path)
        
    else:
        eeg_sessions = []
        effective_fs = None

        # -----------------------------------------------------
        # 3. Batch Preprocessing with Per-File Caching
        # -----------------------------------------------------
        for file_path in nwb_files:
            session_id = file_path.stem
            cached_env_file = cache_dir / f"{session_id}_{target_electrode}_env.npy"

            if cached_env_file.exists():
                print(f"[{target_electrode}] Loading cached envelopes: {cached_env_file.name}")
                smoothed_matrix = np.load(cached_env_file)
                
                if effective_fs is None:
                    # Grab sampling rate to ensure downstream continuity
                    _, effective_fs = load_eeg_trace(file_path, electrode_label=target_electrode)
                    
                eeg_sessions.append(smoothed_matrix)
            else:
                print(f"[{target_electrode}] Preprocessing raw trace: {file_path.name}")
                try:
                    # Extracts, maps to CCF (MOs), and decimates to 500 Hz
                    raw_eeg, fs = load_eeg_trace(file_path, electrode_label=target_electrode)
                    effective_fs = fs

                    notched_eeg = apply_notch(raw_eeg, fs)
                    smoothed_matrix = smooth_env(notched_eeg, fs, trim_edges=True)
                    
                    # Persist matrix to disk immediately
                    np.save(cached_env_file, smoothed_matrix)
                    print(f"  -> Cached envelope matrix to: {cached_env_file.name}")
                    eeg_sessions.append(smoothed_matrix)
                    
                except Exception as e:
                    print(f"  -> Error processing {file_path.name} for {target_electrode}: {e}")
                finally:
                    gc.collect()

        if not eeg_sessions or effective_fs is None:
            print(f"[{target_electrode}] No sessions were successfully processed. Terminating.")
            return

        # -----------------------------------------------------
        # 4. Motif Discovery & Checkpoint 2
        # -----------------------------------------------------
        print(f"\n[{target_electrode}] Discovering motifs across {len(eeg_sessions)} sessions...")
        print(f"Sampling frequency: {effective_fs:.2f} Hz | Null shifts: 20")

        final_motifs = get_motifs(eeg_sessions, effective_fs, num_shifts=20)

        # Free massive session matrices immediately
        del eeg_sessions
        gc.collect()

        # Save motif matrix to disk
        np.save(motif_out_path, final_motifs)
        print(f"[{target_electrode}] Saved recovered motifs to: {motif_out_path.name}")

    # ---------------------------------------------------------
    # 5. Visualization & Export
    # ---------------------------------------------------------
    if final_motifs.shape[1] == 0:
        print(f"[{target_electrode}] No motifs survived null-hypothesis filtering and PCA consolidation.")
        return

    n_motifs = final_motifs.shape[1] // 2
    print(f"[{target_electrode}] Rendering {n_motifs} opponent motif pair(s)...")

    fig, axes = plt.subplots(n_motifs, 1, figsize=(10, 4 * n_motifs), squeeze=False)
    x_freqs = np.arange(1, 77)

    for i in range(n_motifs):
        ax = axes[i, 0]
        ax.plot(x_freqs, final_motifs[:, i], label=f'Motif {i+1}A', color='navy', linewidth=2)
        ax.plot(x_freqs, final_motifs[:, i + n_motifs], label=f'Motif {i+1}B', color='crimson', linewidth=2, linestyle='--')
        
        ax.set_title(f"Opponent Motif Pair {i+1} ({target_electrode})")
        ax.set_xlabel("Frequency Band Starting Frequency (Hz)")
        ax.set_ylabel("Mean Signed BCM Value")
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_out_path, dpi=300)
    print(f"[{target_electrode}] Saved figure to: {fig_out_path}")
    plt.show()

if __name__ == "__main__":
    main()