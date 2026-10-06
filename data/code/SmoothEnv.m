function smoothLFPNorm = SmoothEnv(LFP,LFPfs,freqBands)
    % SMOOTHENV extracts smoothed, normalized amplitude envelopes from LFP
    % across a user-defined set of frequency bands.
    %
    % INPUTS
    % LFP:
    %   Single continuous LFP recording. Input may be a row or column vector.
    %
    % LFPfs:
    %   Sampling rate of the LFP in Hz.
    %
    % freqBands:
    %   N x 2 matrix specifying the lower and upper frequency limits of each
    %   frequency band:
    %
    %       [lowFreq1 highFreq1
    %        lowFreq2 highFreq2
    %        ...
    %        lowFreqN highFreqN]
    %
    % OUTPUT
    % smoothLFPNorm:
    %   Time x frequency-band matrix containing the smoothed Hilbert amplitude
    %   envelope for each frequency band, z-scored across time within each band.
    %
    % PROCESSING
    %   1. Bandpass filter the LFP separately for each requested frequency band.
    %   2. Calculate the Hilbert amplitude envelope of every filtered band.
    %   3. Smooth amplitude envelopes using a 5-second Gaussian window.
    %   4. Remove the first and last 10 seconds to avoid filtering edge artifacts.
    %   5. Z-score each frequency-band amplitude envelope across time while
    %      ignoring the NaN edge regions.
    
    
    % Force the input LFP into a column vector so time is represented along
    % the first dimension throughout the function.
    LFP = LFP(:);
    
    % Number of frequency bands to extract.
    nBands = size(freqBands,1);
    
    % Preallocate a time x frequency-band matrix for the bandpass-filtered LFP.
    filtLFP = zeros(length(LFP),nBands);
    
    % Filter the original LFP into each requested frequency band.
    % filtfilt performs zero-phase filtering to avoid introducing a temporal
    % phase shift into the signal.
    for f = 1:nBands
        [SOS,G] = butterfilter(LFPfs,freqBands(f,1),freqBands(f,2));
        filtLFP(:,f) = filtfilt(SOS,G,double(LFP));
    end
    
    % Calculate the analytic signal with the Hilbert transform and retain its
    % magnitude, giving the instantaneous amplitude envelope of each band.
    % hilbert operates independently on each frequency-band column.
    envLFP = abs(hilbert(filtLFP));
    
    % Duration of Gaussian smoothing applied to the amplitude envelopes.
    LFPSmoothTime = 5;
    
    % Construct a 5-second Gaussian smoothing kernel at the LFP sampling rate.
    w_gamma = gausswin(round(LFPSmoothTime*LFPfs),1);
    
    % Normalize the kernel so smoothing does not change the overall signal scale.
    w_gamma = w_gamma/sum(w_gamma);
    
    % Smooth the amplitude envelopes for all frequency bands.
    % filtfilt applies the Gaussian kernel forward and backward to avoid
    % introducing a temporal shift.
    smoothLFPNorm = filtfilt(w_gamma,1,envLFP);
    
    % Convert the 10-second edge exclusion period into samples.
    edgeSamp = round(10*LFPfs);
    
    % Replace the beginning and end of the recording with NaNs to exclude
    % regions most susceptible to filtering and Hilbert-transform edge artifacts.
    smoothLFPNorm(1:edgeSamp,:) = nan;
    smoothLFPNorm(end-edgeSamp+1:end,:) = nan;
    
    % Calculate the mean and standard deviation of each frequency-band envelope
    % across time while excluding the NaN edge regions.
    mu = mean(smoothLFPNorm,1,'omitnan');
    sigma = std(smoothLFPNorm,0,1,'omitnan');
    
    % Z-score each frequency-band amplitude envelope independently across time.
    % Positive and negative values therefore indicate amplitude above or below
    % that band's session mean, respectively.
    smoothLFPNorm = (smoothLFPNorm-mu)./sigma;
end