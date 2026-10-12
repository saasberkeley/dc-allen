import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

def generate_shifts(n_bands, fs, n_shifts=20):
    """
    Generates independent circular shifts in multiples of 20 seconds.
    The first column is strictly zeros (representing the real, unshifted data).
    """
    shift_times = np.round(fs * 20.0 * np.arange(n_bands)).astype(int)
    
    shifts = np.zeros((n_bands, 100), dtype=int)
    for i in range(100):
        shifts[:, i] = np.random.permutation(shift_times)
        
    shifts = np.unique(shifts, axis=1)
    
    n_available = min(n_shifts, shifts.shape[1])
    shifts = shifts[:, :n_available]
    
    final_shifts = np.zeros((n_bands, n_available + 1), dtype=int)
    final_shifts[:, 1:] = shifts
    return final_shifts

def _get_tree_paths(Z, num_leaves):
    """
    Reconstructs the hierarchical path for every time point.
    Returns lists formatted as [leaf, parent1, parent2, ..., root]
    """
    paths = {i: [i] for i in range(num_leaves)}
    
    for i, merge in enumerate(Z):
        cluster_id = num_leaves + i
        child1, child2 = int(merge[0]), int(merge[1])
        
        for leaf in paths:
            if paths[leaf][-1] in (child1, child2):
                paths[leaf].append(cluster_id)
                
    return [paths[i] for i in range(num_leaves)]

def get_motifs(lfp_sessions, fs, num_shifts=20, corr_thresh=np.arange(0.4, 0.9, 0.1)):
    """
    Identifies recurring coordinated multi-frequency patterns across recording sessions.
    """
    hop = int(np.round(fs))
    n_bands = lfp_sessions[0].shape[0]
    
    shifts = generate_shifts(n_bands, fs, num_shifts)
    n_total_shifts = shifts.shape[1]
    
    final_motifs_all = [[[] for _ in range(len(corr_thresh))] for _ in range(n_total_shifts)]
    
    all_ind_raw = np.concatenate([np.full(session.shape[1], i) for i, session in enumerate(lfp_sessions)])
    all_ind_sampled = all_ind_raw[::hop]
    
    real_all_corr = None
    real_current_inds = None
    
    for s in range(n_total_shifts):
        shifted_sessions = []
        for session in lfp_sessions:
            shifted_session = np.zeros_like(session)
            for f in range(n_bands):
                shifted_session[f, :] = np.roll(session[f, :], shifts[f, s])
            shifted_sessions.append(shifted_session)
            
        all_norm = np.concatenate(shifted_sessions, axis=1)
        all_norm = all_norm[:, ::hop]
        
        all_corr = np.sign(all_norm)
        
        valid_cols = np.abs(np.sum(all_corr, axis=0)) != n_bands
        all_corr = all_corr[:, valid_cols]
        current_inds = all_ind_sampled[valid_cols] if s == 0 else None
        
        if all_corr.shape[1] < 2:
            continue
            
        dist = pdist(all_corr.T, metric='correlation')
        Z = linkage(dist, method='average')
        orig_links = _get_tree_paths(Z, all_corr.shape[1])
        
        for t_idx, thresh in enumerate(corr_thresh):
            links = [list(path) for path in orig_links]
            motif_track = [[i] for i in range(len(links))]
            
            converged = False
            while not converged:
                n_start = len(links)
                new_links = []
                new_track = []
                
                # Fetch immediate parent (index 1) for bottom-up traversal
                sec_links = [path[1] if len(path) > 1 else -1 for path in links]
                
                parent_map = {}
                for i, p in enumerate(sec_links):
                    if p != -1:
                        parent_map.setdefault(p, []).append(i)
                        
                processed = set()
                for parent, indices in parent_map.items():
                    if len(indices) > 1:
                        mean_patterns = np.column_stack([np.mean(all_corr[:, motif_track[idx]], axis=1) for idx in indices])
                        proj = all_corr.T @ mean_patterns
                        
                        std_proj = np.std(proj, axis=0)
                        if np.any(std_proj == 0):
                            r = 0 
                        else:
                            r = np.corrcoef(proj[:, 0], proj[:, 1])[0, 1]
                            
                        if r >= thresh:
                            merged_track = []
                            merged_path_raw = []
                            for idx in indices:
                                merged_track.extend(motif_track[idx])
                                merged_path_raw.extend(links[idx])
                                
                            new_track.append(list(set(merged_track)))
                            
                            # Strip the child nodes from the new path (matches MATLAB tempLinks(1:length) = [])
                            merged_path = sorted(list(set(merged_path_raw)))
                            merged_path = merged_path[len(indices):]
                            
                            new_links.append(merged_path)
                            processed.update(indices)
                            
                for i in range(len(links)):
                    if i not in processed:
                        new_links.append(links[i])
                        new_track.append(motif_track[i])
                        
                # Collapse duplicate hierarchical paths
                unique_mapping = {}
                for i, path in enumerate(new_links):
                    path_tuple = tuple(sorted(path))
                    if path_tuple not in unique_mapping:
                        unique_mapping[path_tuple] = (path, new_track[i])
                        
                links = [v[0] for v in unique_mapping.values()]
                motif_track = [v[1] for v in unique_mapping.values()]
                
                if len(links) == n_start:
                    converged = True

            final_motifs_all[s][t_idx] = motif_track

        if s == 0:
            real_all_corr = all_corr
            real_current_inds = current_inds

    # Null comparison filtering
    all_motifs = final_motifs_all[0]
    
    max_null_size = np.zeros(len(corr_thresh), dtype=int)
    for t_idx in range(len(corr_thresh)):
        sizes = [len(m) for s in range(1, n_total_shifts) for m in final_motifs_all[s][t_idx]]
        if sizes:
            max_null_size[t_idx] = max(sizes)
            
    n_sessions = len(lfp_sessions)
    surviving_motifs = []
    
    for t_idx in range(len(corr_thresh)):
        for motif_indices in all_motifs[t_idx]:
            if len(motif_indices) > max_null_size[t_idx]:
                sessions_present = len(np.unique(real_current_inds[motif_indices]))
                if sessions_present == n_sessions:
                    surviving_motifs.append(motif_indices)
                    
    if not surviving_motifs:
        return np.zeros((n_bands, 0))
        
    surviving_templates = np.column_stack([np.mean(real_all_corr[:, indices], axis=1) for indices in surviving_motifs])
    
    # Uncentered PCA for redundancy consolidation
    U, s_vals, Vt = np.linalg.svd(surviving_templates.T, full_matrices=False)
    coeff = Vt.T
    score = U * s_vals
    mag_score = np.abs(score)
    
    max_comp_assignments = np.argmax(mag_score, axis=1)
    unique_comps, counts = np.unique(max_comp_assignments, return_counts=True)
    valid_comps = unique_comps[counts > 1]
    
    if len(valid_comps) == 0:
        return np.zeros((n_bands, 0))
        
    retained_coeff = coeff[:, valid_comps]
    proj = real_all_corr.T @ retained_coeff
    proj_concat = np.hstack([proj, -proj])
    
    n_valid = len(valid_comps)
    final_motifs = np.zeros((n_bands, n_valid))
    
    for i in range(n_valid):
        max_vals = np.max(proj_concat, axis=1)
        pos_mask = (proj_concat[:, i] == max_vals)
        neg_mask = (proj_concat[:, i + n_valid] == max_vals)
        
        pos_samples = real_all_corr[:, pos_mask]
        neg_samples = -real_all_corr[:, neg_mask]
        
        combined_samples = np.concatenate([pos_samples, neg_samples], axis=1)
        if combined_samples.shape[1] > 0:
            final_motifs[:, i] = np.mean(combined_samples, axis=1)
            
    final_motifs_paired = np.hstack([final_motifs, -final_motifs])
    return final_motifs_paired