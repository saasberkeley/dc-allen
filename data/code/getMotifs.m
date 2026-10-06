function finalMotifs = getMotifs(LFPIn,LFPfs)

% GETMOTIFS identifies recurring coordinated multi-frequency patterns
% across recording sessions.
%
% INPUTS
% LFPIn:
%   Column cell-array vector with one cell for each rat/session.
%   Each cell contains a matrix of:
%
%       frequency bands x time samples
%
%   All sessions should contain the same number and ordering of frequency
%   bands, although recording duration can differ across sessions.
%
%   The algorithm ultimately binarizes the input using sign(), so each
%   frequency band should already be z-scored.
%
% LFPfs:
%   Sampling rate (Hz) of the time dimension in LFPIn.
%
% OUTPUT
% finalMotifs:
%   Matrix containing the final spectral motif templates:
%
%       frequency bands x motif states
%
%   The first half contains the identified motifs and the second half
%   contains their exact sign-inverted counterparts. Thus, each motif is
%   explicitly represented by positive and negative expression states.
%
% OVERVIEW
%   1. Sample the data at approximately 1-second intervals.
%   2. Binarize each frequency band according to its sign.
%   3. Hierarchically cluster recurring multi-frequency sign patterns.
%   4. Iteratively consolidate branches whose motif projection time courses
%      are sufficiently correlated.
%   5. Repeat across several consolidation thresholds.
%   6. Generate null datasets by independently circularly shifting frequency
%      bands relative to one another.
%   7. Retain empirical motifs that are larger than motifs observed in the
%      shifted nulls and that are represented in every recording session.
%   8. Consolidate redundant motifs recovered across thresholds using PCA.
%   9. Re-estimate the final motif templates from the original unshifted
%      data and append their sign-inverted counterparts.


    % Analyze approximately one sample per second.
    hop = round(LFPfs);

    % Create a session identity vector for every time point so that the
    % sessions contributing to each motif can later be determined.
    allInd = cell(size(LFPIn,1),1);

    for i = 1:size(allInd,1)

        allInd{i} = ones(1,size(LFPIn{i},2))*i;

    end

    allInd = cell2mat(allInd(:)');

    % Match the temporal sampling used below for motif detection.
    allInd = allInd(1:hop:end);

    freqsOfInt = 1:size(LFPIn{1,1},1);

    % Generate possible circular shifts in 20-second increments.
    % Each null realization independently shifts the frequency bands relative
    % to one another, disrupting their true temporal co-fluctuation while
    % preserving the structure within each individual frequency band.
    shiftTimes = round(LFPfs*20.*(0:(size(LFPIn{1},1)-1)));

    numShifts = 20;

    shifts = zeros(length(freqsOfInt),100);

    for i = 1:100

        shifts(:,i) = shiftTimes(randperm(length(shiftTimes)));

    end

    % Remove duplicate random shift configurations.
    shifts = unique(shifts','rows')';

    % Column 1 is the original unshifted data; the remaining columns are
    % independently shifted null realizations.
    shifts = [zeros(length(freqsOfInt),1),shifts(:,1:numShifts)];

    % Candidate thresholds used when determining whether nearby branches of
    % the hierarchical tree represent sufficiently similar motif dynamics to
    % be consolidated.
    corrThresh = 0.4:0.1:0.8; % corr thresh for consolidation

    % Stores motif membership for the real data and every shifted null at
    % each consolidation threshold.
    finalMotifsAll = cell(numShifts+1,length(corrThresh));

    %%

    % Run the complete clustering procedure first on the real data and then
    % on each independently shifted null realization.
    for s = 1:size(shifts,2)

        allNorm = LFPIn(:);

        % Apply the frequency-specific circular shifts for this realization.
        % For the first realization all shifts are zero.
        for sc = 1:length(allNorm)

            for f = 1:size(shifts,1)

                allNorm{sc}(f,:) = circshift(allNorm{sc}(f,:),shifts(f,s));

            end

        end

        % Concatenate all recording sessions along time.
        allNorm = cell2mat(allNorm');

        % Sample approximately once per second.
        allNorm = allNorm(:,1:hop:end);

        % Reduce each time point to its coordinated above/below-baseline
        % pattern across frequencies. Amplitude magnitude is therefore not
        % used to define motif membership.
        allCorr = sign(allNorm);

        % Remove time points in which every frequency has the same sign.
        % These correspond to trivial globally high or globally low patterns
        % rather than structured relationships across frequencies.
        allCorr(:,abs(sum(allCorr))==size(allCorr,1))=[];

        % Compute correlation distance between the multi-frequency pattern
        % observed at every retained time point.
        D = pdist(allCorr', 'correlation');  % 

        % Step 2: Perform hierarchical clustering

        % Average-linkage clustering provides the initial hierarchy of
        % recurring spectral configurations.
        Z = linkage(D, 'average');  %

        % Track the complete path through the hierarchical tree for every
        % original time point.
        links = num2cell(1:size(allCorr,2))';

        numIn = size(allCorr,2);

        while 1

            stopTally = 0;

            stopAll = 0;

            for i = 1:size(allCorr,2)

                [rLink,~] = find(Z(:,1:2)==links{i}(end));

                if ~isempty(rLink)

                    links{i} = [links{i},numIn+rLink];

                end

                if isempty(rLink)

                    stopTally = stopTally + 1;

                end

                if stopTally == numIn

                    stopAll = 1;

                    break

                end

            end

            if stopAll == 1

                break

            end

        end

        % Preserve the original tree paths so that the same hierarchy can be
        % evaluated independently at each correlation threshold.
        origLinks = links';

        %

        % Track which original time points currently contribute to each
        % candidate motif.
        finalMotifsTrackOrig = num2cell(1:length(links));

        links = origLinks;

        % Repeat motif consolidation across the range of correlation
        % thresholds.
        for t = 1:length(corrThresh)

            disp([num2str(s),'_',num2str(t)])

            corrMotifThresh = corrThresh(t);

            links = origLinks;

            finalMotifsTrack = finalMotifsTrackOrig;

            outterLinkSize = [];

            linkSize = length(links);

            n = 1;

            % Iteratively consolidate nearby branches until no additional
            % motifs can be merged at the current threshold.
            while 1

                finalMotifsStore = [];

                newLinksStore = [];

                m = 1;

                while 1 

                    newLinks = cell(size(links));

                    lowCorr = zeros(size(links));

                    % The second element identifies the first shared branch
                    % point in the current hierarchical representation.
                    secLink = cellfun(@(x) x(2),links);

                    for i = 1:length(links)

                        targetLink = links{i}(2);

                        getMatches = find(targetLink==secLink);

                        if length(getMatches)>1

                            % Calculate the mean spectral pattern represented
                            % by each of the paired branches.
                            tempMotifs = zeros(size(allCorr,1),size(getMatches,1));

                            for j = 1:length(getMatches)

                                tempMotifs(:,j) = mean(allCorr(:,finalMotifsTrack{getMatches(j)}),2);

                            end

                            % Project the complete dataset onto the candidate
                            % motifs. Branches are merged according to the
                            % similarity of their expression over time rather
                            % than coefficient similarity alone.
                            projVec = allCorr'*tempMotifs;

                            r = corr(projVec(:,1),projVec(:,2));

                        end

                        if length(getMatches)>1

                            if r>=corrMotifThresh

                                % Candidate branches with sufficiently
                                % correlated temporal expression are merged.
                                tempLinks = unique(cell2mat(links(getMatches))); tempLinks(1:length(getMatches))=[];

                                tempMotifTrack = cell2mat(finalMotifsTrack(getMatches));

                                for j = 1:length(getMatches)

                                    newLinks{getMatches(j)} = tempLinks;

                                    finalMotifsTrack{getMatches(j)} = tempMotifTrack;

                                end

                            else

                                % Branches below the consolidation threshold
                                % are retained as separate motif candidates
                                % and removed from further merging.
                                for j = 1:length(getMatches)

                                    newLinks{getMatches(j)} = links{getMatches(j)};

                                    lowCorr(getMatches(j)) = 1;

                                end

                            end

                        else

                            newLinks{getMatches} = links{getMatches}; 

                        end

                    end

                    % Remove duplicate sample identities created by merging.
                    finalMotifsTrack = cellfun(@unique,finalMotifsTrack,'UniformOutput',0);

                    % Temporarily store branches that failed the correlation
                    % criterion so that they are preserved as independent
                    % motif candidates.
                    finalMotifsStore = [finalMotifsStore,finalMotifsTrack(lowCorr==1)]; finalMotifsTrack(lowCorr==1)=[];

                    newLinksStore = [newLinksStore,newLinks(lowCorr==1)]; newLinks(lowCorr==1)=[];

                    % Collapse duplicate hierarchical paths produced by the
                    % same merge.
                    linkKeys = string(cellfun(@(x) strjoin(string(sort(x)), ','), ...
                          newLinks, 'UniformOutput', false));

                    [~, ia] = unique(linkKeys, 'stable'); 

                    links = newLinks(ia);

                    finalMotifsTrack = finalMotifsTrack(ia);

                    linkSize = [linkSize;length(links)];

                    % Stop the current consolidation pass when the number of
                    % remaining branches no longer changes.
                    if m > 1 && linkSize(end)==linkSize(end-1)

                        finalMotifsTrack = [finalMotifsTrack,finalMotifsStore];

                        links = [links,newLinksStore];

                        break

                    else

                        m = m + 1;

                    end

                end

                outterLinkSize = [outterLinkSize;length(links)];

                % Convert the current sample memberships into mean
                % multi-frequency motif templates.
                finalMotifs = cell2mat(cellfun(@(x) mean(allCorr(:,x),2),finalMotifsTrack,'UniformOutput',0));

                % Once a complete re-clustering pass no longer changes the
                % number of motifs, consolidation has converged.
                if n > 1 && outterLinkSize(end)-outterLinkSize(end-1)==0

                    break

                else

                    % Re-cluster the newly consolidated motif templates and
                    % repeat the same branch-merging procedure. This allows
                    % motifs that become similar only after earlier
                    % consolidation steps to subsequently merge.
                    DD = pdist(finalMotifs', 'correlation');  % 

                    % Step 2: Perform hierarchical clustering

                    ZZ = linkage(DD, 'average');  %

                    links = num2cell(1:size(finalMotifs,2));

                    numIn = size(finalMotifs,2);

                    % Reconstruct the complete hierarchical paths for the
                    % newly consolidated motif templates.
                    while 1

                        stopTally = 0;

                        stopAll = 0;

                        for i = 1:size(finalMotifs,2)

                            [rLink,~] = find(ZZ(:,1:2)==links{i}(end));

                            if ~isempty(rLink)

                                links{i} = [links{i},numIn+rLink];

                            end

                            if isempty(rLink)

                                stopTally = stopTally + 1;

                            end

                            if stopTally == numIn

                                stopAll = 1;

                                break

                            end

                        end

                        if stopAll == 1

                            break

                        end

                    end

                    n = n + 1;

                end

            end

            % Store the original time-point memberships of every motif
            % recovered for this shift realization and threshold.
            finalMotifsAll{s,t} = finalMotifsTrack;

        end

    end

    % Reconstruct the original, unshifted binarized dataset. All subsequent
    % filtering and final motif estimation are performed on the real data.
    allNorm = LFPIn(:);

    allNorm = cell2mat(allNorm');

    allNorm = allNorm(:,1:hop:end);

    allCorr = sign(allNorm);

    % Keep the session identity vector aligned with the retained time points.
    allInd(abs(sum(allCorr))==size(allCorr,1))=[];

    allCorr(:,abs(sum(allCorr))==size(allCorr,1))=[];

    %% null compare

    % For each consolidation threshold, determine the largest motif produced
    % by any independently shifted null dataset.
    maxMot = zeros(size(finalMotifsAll));

    for i = 2:size(finalMotifsAll,1)

        for j = 1:size(finalMotifsAll,2)

            maxMot(i,j) = max(cellfun(@length,finalMotifsAll{i,j}));

        end

    end

    % Null criterion for each consolidation threshold: empirical motifs must
    % contain more samples than the largest motif recovered from any shifted
    % realization.
    maxMot = max(maxMot);

    % The first row contains motifs recovered from the real, unshifted data.
    allMotifs = finalMotifsAll(1,:); 

    finalMotifs = cell(size(allMotifs));

    allMotifInd = cell(size(allMotifs));

    for i = 1:length(allMotifs)

        sizeMot = cellfun(@length,allMotifs{i});

        % Retain only empirical motifs larger than the maximum corresponding
        % shifted-null motif.
        allMotifs{i} = allMotifs{i}(sizeMot>maxMot(i));

        % Calculate the mean spectral template of each surviving motif.
        finalMotifs{i} = cell2mat(cellfun(@(x) mean(allCorr(:,x),2),allMotifs{i},'UniformOutput',0));

        % Determine which recording sessions contributed samples to each
        % motif.
        allMotifInd{i} = cellfun(@(x) unique(allInd(x)),allMotifs{i},'UniformOutput',0);

        % Require every retained motif to be represented in every input
        % recording session.
        allMotifs{i} = allMotifs{i}(cellfun(@length,allMotifInd{i})==length(LFPIn));

        finalMotifs{i} = finalMotifs{i}(:,cellfun(@length,allMotifInd{i})==length(LFPIn));

        % Alternative criterion retained from development: require motif
        % representation in more than one session rather than all sessions.
        % allMotifs{i} = allMotifs{i}(cellfun(@length,allMotifInd{i})>1);

        % finalMotifs{i} = finalMotifs{i}(:,cellfun(@length,allMotifInd{i})>1);

    end

    % Pool the sample memberships of motifs surviving across all tested
    % consolidation thresholds.
    tempMotifs = [];

    for i = 1:length(allMotifs)

        tempMotifs = [tempMotifs,allMotifs{i}];

    end

    allMotifs = tempMotifs;

    % Pool the corresponding motif templates across thresholds.
    finalMotifs = cell2mat(finalMotifs);

    % Use uncentered PCA to consolidate highly redundant motif templates
    % recovered independently across different consolidation thresholds.
    [coeff,score,~] = pca(finalMotifs','centered','off');

    magScore = abs(score);

    % Assign every candidate motif to the PCA component on which it has the
    % largest absolute score.
    [rmax,cmax] = find(magScore==repmat(max(magScore,[],2),[1,size(magScore,2)]));

    findMax = sortrows([rmax,cmax]); findMax = findMax(:,2);

    % Retain PCA components supported by more than one independently
    % recovered candidate motif.
    numMot = histcounts(findMax,0.5:size(coeff,2)+0.5); numMot = find(numMot>1);

    % Project every original binarized time point onto the retained motif
    % axes and their sign-inverted counterparts.
    proj = allCorr'*coeff(:,numMot); proj = [proj,-proj];

    finalMotifs = zeros(size(finalMotifs,1),length(numMot));

    for i = 1:length(numMot)

        % Re-estimate each final motif directly from the original data.
        % Samples maximally expressing the positive pole are averaged with
        % the sign-inverted samples maximally expressing its negative pole,
        % producing a single canonical motif orientation.
        finalMotifs(:,i) = mean([allCorr(:,proj(:,i)==max(proj,[],2)),-allCorr(:,proj(:,i+length(numMot))==max(proj,[],2))],2);

    end

    % Explicitly append the sign-inverted opponent of every final motif.
    finalMotifs = [finalMotifs,-finalMotifs];

end