import numpy as np
from scipy import signal
from scipy.fft import next_fast_len
from pynwb import NWBHDF5IO

# Define the 76 overlapping 5 Hz frequency bands (1-5 Hz up to 76-80 Hz)
FREQ_BANDS = np.column_stack([np.arange(1, 77), np.arange(5, 81)])

# Equivalent mapping dictionary
ELECTRODE_MAP = {
    'Fz': 'MOs',
    'Cz': 'MOp',
    'Pz': 'RSPagl'
}

def load_eeg_trace(nwb_path, electrode_label='Fz', target_fs=500.0):
    """
    Extracts a 1D EEG time series for a specified electrode.
    Accepts 10-20 proxies ('Fz', 'Cz', 'Pz') which map to CCF names ('MOs', 'MOp', 'RSPagl'),
    or exact CCF location names directly. Decimates to target_fs to improve performance.
    """
    target = ELECTRODE_MAP.get(electrode_label, electrode_label)
    
    with NWBHDF5IO(nwb_path, mode='r') as io:
        nwbfile = io.read()
        eeg_series = nwbfile.acquisition['ElectricalSeriesEEG']
        
        # Correctly handle the sampling rate
        if eeg_series.rate is not None:
            raw_fs = float(eeg_series.rate)
        else:
            raw_fs = float(1.0 / np.median(np.diff(eeg_series.timestamps[:1000])))
            
        electrodes_df = nwbfile.electrodes.to_dataframe()
        eeg_table = electrodes_df[electrodes_df['group_name'] == 'EEG array']
        
        matched_indices = eeg_table.index[eeg_table['location'] == target].tolist()
        if not matched_indices:
            raise ValueError(
                f"Electrode '{target}' not found. Available locations: {eeg_table['location'].unique().tolist()}"
            )
            
        ch_idx = matched_indices[0]
        eeg_data = np.asarray(eeg_series.data[:, ch_idx], dtype=np.float64) * float(eeg_series.conversion)
        
    # Downsample if the raw sampling rate is significantly higher than target_fs
    if target_fs is not None and raw_fs > target_fs:
        q = int(round(raw_fs / target_fs))
        actual_fs = raw_fs / q
        # Zero-phase IIR decimation to prevent phase distortion
        eeg_data = signal.decimate(eeg_data, q, ftype='iir', zero_phase=True)
    else:
        actual_fs = raw_fs
        
    return eeg_data, actual_fs

def apply_notch(lfp, fs, low=59.0, high=61.0):
    """
    Applies a 6th-order Butterworth bandstop filter to remove 60 Hz line noise.
    """
    sos = signal.butter(N=6, Wn=[low, high], btype='bandstop', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, lfp)

def smooth_env(lfp, fs, bands=FREQ_BANDS, trim_edges=True):
    """
    Extracts smoothed, normalized amplitude envelopes across frequency bands.
    
    Parameters:
        lfp (np.ndarray): 1D array of the continuous recording.
        fs (float): Sampling frequency in Hz.
        bands (np.ndarray): N x 2 matrix of frequency limits.
        trim_edges (bool): If True, removes the 10-second NaN boundaries.
        
    Returns:
        smooth_lfp_norm (np.ndarray): Matrix of shape (n_bands, n_samples).
    """
    n_samples = len(lfp)
    n_bands = bands.shape[0]
    
    # Pad for efficient FFT computation during the Hilbert transform
    n_fft = next_fast_len(n_samples)
    
    # Create 5-second Gaussian smoothing kernel
    smooth_time = 5.0
    M = int(round(smooth_time * fs))
    std = (M - 1) / 2.0
    w_gamma = signal.windows.gaussian(M, std=std)
    w_gamma /= w_gamma.sum()
    
    # Preallocate output matrix: (Frequency Bands x Time)
    smooth_lfp_norm = np.zeros((n_bands, n_samples), dtype=np.float32)
    
    for i, (low, high) in enumerate(bands):
        # 1. Bandpass filter (6th order Butterworth zero-phase)
        sos = signal.butter(N=6, Wn=[low, high], btype='bandpass', fs=fs, output='sos')
        filt_lfp = signal.sosfiltfilt(sos, lfp)
        
        # 2. Hilbert amplitude envelope
        # n_fft padding prevents scipy.signal.hilbert from hanging on 11.5M samples
        analytic_signal = signal.hilbert(filt_lfp, N=n_fft)[:n_samples]
        env_lfp = np.abs(analytic_signal)
        
        # 3. Gaussian smoothing
        smooth_env_band = signal.filtfilt(w_gamma, [1.0], env_lfp)
        
        # Store in matrix
        smooth_lfp_norm[i, :] = smooth_env_band

    # 4. Handle edge artifacts
    edge_samp = int(round(10.0 * fs))
    smooth_lfp_norm[:, :edge_samp] = np.nan
    smooth_lfp_norm[:, -edge_samp:] = np.nan
    
    # 5. Z-score each frequency band across time (ignoring NaNs)
    mu = np.nanmean(smooth_lfp_norm, axis=1, keepdims=True)
    sigma = np.nanstd(smooth_lfp_norm, axis=1, ddof=1, keepdims=True)
    smooth_lfp_norm = (smooth_lfp_norm - mu) / sigma
    
    # 6. Drop the NaN edges completely to avoid breaking downstream clustering
    if trim_edges:
        # Slice first so normalization is computed strictly on retained points
        smooth_lfp_norm = smooth_lfp_norm[:, edge_samp:-edge_samp]
        mu = np.mean(smooth_lfp_norm, axis=1, keepdims=True)
        sigma = np.std(smooth_lfp_norm, axis=1, ddof=1, keepdims=True)
        smooth_lfp_norm = (smooth_lfp_norm - mu) / sigma
    else:
        # Keep MATLAB-style NaN boundary padding
        smooth_lfp_norm[:, :edge_samp] = np.nan
        smooth_lfp_norm[:, -edge_samp:] = np.nan
        mu = np.nanmean(smooth_lfp_norm, axis=1, keepdims=True)
        sigma = np.nanstd(smooth_lfp_norm, axis=1, ddof=1, keepdims=True)
        smooth_lfp_norm = (smooth_lfp_norm - mu) / sigma
        
    return smooth_lfp_norm