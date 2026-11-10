function allClust = getClust(X, LFPfs)
% GETCLUST Identify groups of neurons with coordinated activity epochs.
%
%   allClust = getClust(X, LFPfs)
%
% INPUTS
%   X       - [time samples x neurons] matrix of PREPROCESSED cellular
%             activity. The manuscript describes squaring and z-scoring
%             each neuron's continuous rate trace. Those operations must
%             occur upstream: this function does NOT perform them.
%   LFPfs   - Sampling rate, in Hz, of the rows of X. It converts the
%             minimum event duration (3 s) into a number of samples.
%
% OUTPUT
%   allClust - Cell array; each entry lists the column indices of neurons
%             assigned to one retained cluster. Neurons left unassigned do
%             not appear in the output.
%
% OVERVIEW
%   1. Propose candidate groups using activity-defined co-recruitment epochs
%      and iteratively remove weakly recruited neurons.
%   2. Test each candidate with 500 random, three-neuron coalitions; assess
%      every OUT-OF-COALITION neuron against events defined by the coalition.
%   3. Retain reliable groups (at least five neurons), remove their members
%      from the available pool, and repeat to find additional groups.
%   4. Merge groups showing sufficiently strong cross-group recruitment.
%
% IMPORTANT DETAILS
%   - Random coalitions are sampled with randperm; no RNG seed is set here.
%     Identical X can produce slightly different results across runs.
%   - A group's support is (fraction of cells with X > 0) * mean(X), NOT
%     the sum of events or a correlation coefficient.
%   - Merging requires strong recruitment in EITHER direction (A -> B OR
%     B -> A); connected components also merge chains transitively.
%   - The function relies on bwconncomp (Image Processing Toolbox).
%
% All algorithmic statements below describe the existing implementation;
% no computational steps or thresholds have been changed.

    % Initially every neuron is eligible to join a candidate cluster.
    allCellsFull = 1:size(X,2);
    remainingCells = allCellsFull;

    allClust = {};

    % Fixed algorithm parameters.
    nSub        = 500;              % Random coalitions per candidate group
    subSize     = 3;                % Neurons in each coalition
    minEventDur = round(3*LFPfs);   % Minimum contiguous event length: 3 s
    supportZ    = 1;                % Support threshold: mean + 1 SD
    scoreThresh = 1;                % Minimum held-out reliability score
    mergeScoreThresh = 0.6;        % Mean cross-group recruitment threshold
    minClustSize = 5;               % Minimum accepted cluster membership

    % Repeat the full candidate-finding and validation pipeline after each
    % accepted batch of neurons is removed from the unassigned pool.
    while ~isempty(remainingCells)

        remainingPrevOuter = remainingCells;

        % Store all provisional groups found in this outer iteration.
        allMasterGroupKeep = {};
        allMasterPcScore = {};  % Legacy/unused storage; has no downstream effect

        masterK = 1;
        seedRemainingCells = remainingCells;

        % Attempt candidate discovery on successive groups of neurons not
        % already accounted for in the current candidate-discovery pass.
        while ~isempty(seedRemainingCells)

            totalKeep = [];
            allGroupKeep = {};
            pcScore = {};  % Legacy/unused variable

            groupKeep = seedRemainingCells;

            k = 1;
            % Keep extracting groups from the unaccounted-for seed cells
            % until all were assigned provisionally or extraction fails.
            while length(totalKeep) ~= length(groupKeep)
                groupIdx = setdiff(groupKeep,totalKeep);

                groupKeepStart = groupIdx(:);
                maxIter = 20;

                % Repeatedly define coactivity epochs for the current group
                % and drop neurons insufficiently recruited during them.
                for it = 1:maxIter
                    oldGroup = groupKeepStart;

                    groupX = X(:,groupKeepStart);

                    % Time-varying group support:
                    %   supportFrac: proportion of neurons with X > 0
                    %   supportAmp:  signed mean activity of group members
                    % Their product requires both prevalent positivity and
                    % a high mean group response.
                    supportFrac = mean(groupX>0,2);
                    supportAmp  = mean(groupX,2);
                    groupSupport = supportFrac .* supportAmp;

                    % Candidate events are high-support time samples.
                    thresh = mean(groupSupport) + supportZ*std(groupSupport);
                    eventMask = groupSupport >= thresh;

                    % Retain only contiguous epochs lasting at least 3 s.
                    cc = bwconncomp(eventMask);
                    eventList = cc.PixelIdxList;
                    eventList(cellfun(@length,eventList) < minEventDur) = [];

                    allEventMask = false(size(groupSupport));
                    for e = 1:length(eventList)
                        allEventMask(eventList{e}) = true;
                    end

                    % Each cell's recruitment = its mean activity during
                    % accepted epochs minus its mean activity outside them.
                    cellScore = zeros(length(groupKeepStart),1);

                    for i = 1:length(groupKeepStart)
                        c = groupKeepStart(i);
                        xc = X(:,c);

                        inScore  = mean(xc(allEventMask));
                        outScore = mean(xc(~allEventMask));

                        cellScore(i) = inScore - outScore;
                    end

                    % Gradually increase the recruitment criterion:
                    % iterations 1,2,3,4+ require > 0.3, 0.6, 0.9, 1.0.
                    keepMask = cellScore>min([1,0.3*it]);

                    groupKeepStart = groupKeepStart(keepMask);

                    % The group has stabilized if no members were removed.
                    if isequal(groupKeepStart,oldGroup)
                        break
                    end
                end

                % A failure to retain any cells ends extraction from this
                % particular set of seed cells.
                if isempty(groupKeepStart)
                    break
                end

                allGroupKeep{k} = groupKeepStart;
                totalKeep = [totalKeep;groupKeepStart];

                k = k + 1;
            end

            if isempty(allGroupKeep)
                break
            end

            allMasterGroupKeep{masterK} = allGroupKeep;
            allMasterPcScore{masterK} = pcScore;  % Not used subsequently

            % Discover additional candidates only among as-yet-unused seed
            % cells; provisional candidates are kept separate for now.
            usedThisPass = unique(cell2mat(allGroupKeep(:)));
            remainingPrev = seedRemainingCells;
            seedRemainingCells = setdiff(seedRemainingCells,usedThisPass);

            if isequal(seedRemainingCells,remainingPrev)
                break
            end

            masterK = masterK + 1;
        end

        % Flatten the nested provisional group collection.
        rawClust = [];
        for i = 1:length(allMasterGroupKeep)
            rawClust = [rawClust,allMasterGroupKeep{i}];
        end

        if isempty(rawClust)
            break
        end

        % Candidate groups must have at least five neurons before the
        % coalition-based reliability check.
        rawClust(cellfun(@length,rawClust)<minClustSize)=[];
        rawClust = rawClust';

        if isempty(rawClust)
            break
        end

        coalitionRank = cell(size(rawClust));
        coalitionScore = cell(size(rawClust));

        % Validate group membership using activity epochs defined from
        % random subsets, so a tested neuron cannot define its own events.
        for cIdx = 1:length(rawClust)

            clustCells = rawClust{cIdx};
            nCells = length(clustCells);

            if nCells < 2
                continue
            end

            groupX = X(:,clustCells);

            % A coalition uses up to three neurons, leaving >=1 held out.
            thisSubSize = min(subSize,nCells-1);

            % Accumulate first/second moments of each held-out neuron's
            % event-versus-nonevent recruitment across valid coalitions.
            scoreSum   = zeros(nCells,1);
            scoreSumSq = zeros(nCells,1);
            validCount = zeros(nCells,1);

            for s = 1:nSub
                subIdx = randperm(nCells,thisSubSize);
                subX = groupX(:,subIdx);

                % Define candidate events using only coalition members.
                supportFrac = mean(subX > 0,2);
                supportAmp  = mean(subX,2);
                subSupport  = supportFrac .* supportAmp;

                thresh = mean(subSupport) + supportZ*std(subSupport);
                eventMask = subSupport >= thresh;

                cc = bwconncomp(eventMask);
                eventList = cc.PixelIdxList;
                eventList(cellfun(@length,eventList) < minEventDur) = [];

                allEventMask = false(size(subSupport));

                for e = 1:length(eventList)
                    allEventMask(eventList{e}) = true;
                end

                % Coalitions producing no valid events contribute no scores.
                if ~any(allEventMask)
                    continue
                end

                testIdx = setdiff(1:nCells,subIdx);

                % Score only the held-out neurons, never coalition members.
                for ii = 1:length(testIdx)

                    i = testIdx(ii);
                    xc = groupX(:,i);

                    inScore  = mean(xc(allEventMask));
                    outScore = mean(xc(~allEventMask));

                    score = inScore - outScore;

                    scoreSum(i)   = scoreSum(i) + score;
                    scoreSumSq(i) = scoreSumSq(i) + score.^2;
                    validCount(i) = validCount(i) + 1;
                end
            end

            % A 'reliability' ratio: mean held-out recruitment divided by
            % its across-coalition SD (plus eps to avoid division by zero).
            % This is not a p-value and does not use an unbiased sample SD.
            meanScore = scoreSum ./ max(validCount,1);

            scoreVar = (scoreSumSq ./ max(validCount,1)) - meanScore.^2;
            scoreStd = sqrt(max(scoreVar,0));

            reliabilityScore = meanScore ./ (scoreStd + eps);

            % Rank candidate members by reliability, retaining the original
            % cell indices from the full matrix X.
            [sortedScore, sortIdx] = sort(reliabilityScore,'descend');

            coalitionRank{cIdx} = clustCells(sortIdx(:));
            coalitionScore{cIdx} = sortedScore(:);
        end

        % Keep only neurons with reliability > 1, then remove clusters
        % whose surviving membership is smaller than five neurons.
        cleanClust = cellfun(@(x,y) x(y>scoreThresh),coalitionRank,coalitionScore,'UniformOutput',0);
        cleanClust(cellfun(@length,cleanClust)<minClustSize)=[];

        if isempty(cleanClust)
            break
        end

        acceptedCells = unique(cell2mat(cleanClust(:)));

        if isempty(acceptedCells)
            break
        end

        % Accept this batch; neurons already accepted are no longer eligible
        % to form additional groups in later outer iterations.
        allClust = [allClust; cleanClust(:)];
        remainingCells = setdiff(remainingCells,acceptedCells);

        if isequal(remainingCells,remainingPrevOuter)
            break
        end
    end

    %% Cross-cluster coalition consolidation
    % Merge retained clusters whose activity co-recruits the OTHER cluster.
    % Crucially, merging is applied after membership selection, not as part
    % of the held-out neuron reliability calculation above.
    didMerge = true;

    while didMerge

        didMerge = false;
        nClust = length(allClust);

        if nClust < 2
            break
        end

        clustEventMask = cell(nClust,1);

        % Define events separately from each accepted cluster's full member
        % set using the same support statistic and duration criterion.
        for cIdx = 1:nClust

            clustCells = allClust{cIdx};
            groupX = X(:,clustCells);

            supportFrac  = mean(groupX > 0,2);
            supportAmp   = mean(groupX,2);
            groupSupport = supportFrac .* supportAmp;

            thresh = mean(groupSupport) + supportZ*std(groupSupport);
            eventMask = groupSupport >= thresh;

            cc = bwconncomp(eventMask);
            eventList = cc.PixelIdxList;
            eventList(cellfun(@length,eventList) < minEventDur) = [];

            allEventMask = false(size(groupSupport));

            for e = 1:length(eventList)
                allEventMask(eventList{e}) = true;
            end

            clustEventMask{cIdx} = allEventMask;
        end

        crossScore = nan(nClust,nClust);

        % Directed cross-group score(a,b): mean across neurons in cluster B
        % of [mean X during A's events - mean X outside A's events].
        for a = 1:nClust

            maskA = clustEventMask{a};

            if ~any(maskA)
                continue
            end

            for b = 1:nClust

                if a == b
                    continue
                end

                cellsB = allClust{b};
                XB = X(:,cellsB);

                cellScores = zeros(length(cellsB),1);

                for i = 1:length(cellsB)

                    xc = XB(:,i);

                    inScore  = mean(xc(maskA));
                    outScore = mean(xc(~maskA));

                    cellScores(i) = inScore - outScore;
                end

                crossScore(a,b) = mean(cellScores);
            end
        end

        % An ordered pair A -> B qualifies when B is sufficiently recruited
        % during A's events. A and B are connected if EITHER direction
        % qualifies. Connected components merge transitively (A-B-C can
        % become one cluster without a direct A-C relationship).
        mergeMat = crossScore > mergeScoreThresh;
        mergeMat(isnan(mergeMat)) = false;
        mergeMat(eye(size(mergeMat)) == 1) = false;

        if ~any(mergeMat(:))
            break
        end

        G = graph(mergeMat | mergeMat');
        compID = conncomp(G);

        newAllClust = {};
        usedComp = unique(compID);

        % Union neuron memberships within each connected component. A
        % singleton component is simply retained as its own cluster.
        for ci = 1:length(usedComp)

            thisComp = usedComp(ci);
            mergeIdx = find(compID == thisComp);

            mergedCells = [];

            for j = 1:length(mergeIdx)
                mergedCells = [mergedCells; allClust{mergeIdx(j)}(:)];
            end

            newAllClust{end+1,1} = unique(mergedCells);
        end

        % Recalculate cluster events/cross-recruitment after every merge
        % round, until no further consolidation occurs.
        if length(newAllClust) < length(allClust)
            allClust = newAllClust;
            didMerge = true;
        end
    end

end
