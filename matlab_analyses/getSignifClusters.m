function [clusPval_Z_Stat, clusPos_Z_Stat, clusPval_clusSize, clusPos_clusSize] = getSignifClusters(p_sig, zscores, p_perm, zscores_perm, THRESH_SUPRACLUSTER, ALPHA)
    % GETSIGNIFCLUSTERS Find significant clusters (in original data) based on 
    % clusters obtained by a permutation procedure
    % (based on Maris & Oostenveld (2007) J Neurosci Meth 164:177-190)
    %
    % INPUTS
    % psig         - significance matrix (FREQ x TIME)
    % zscores      - zscores to divide into positive and negative sig. clusters
    % p_perm       - matrix of p-values derived from permutations
    % zscores_perm - zscores derived from permutation to divide into positive and 
    %                negative sig. clusters (NPERM x FREQ x TIME) where NPERM 
    %                is the number of permutation numPerms
    % THRESH_SUPRACLUSTER - significance threshold to determine supra-threshold 
    %                          clusters (default 0.05)
    % ALPHA - significance threshold, i.e. percentage of the largest 
    %         sums of suprathreshold cluster z-scores (or cluster sizes) from
    %         the permutation distribution that may be larger than the sum from
    %         the original unpermuted data
    %
    % OUTPUTS
    % clusPos_Z_Stat    - Positions of significant clusters based on summed
    %                     z-scores
    % clusPval_Z_Stat   - P-values of clusters based on summed z-scores
    % clusPos_clusSize  - Positions of significant clusters based on cluster size
    % clusPval_clusSize - P-values of clusters based on cluster size, generally provides
    %                     very similar results to clusPval_Z_Stat

    if ~exist('THRESH_SUPRACLUSTER', 'var')
      THRESH_SUPRACLUSTER = 0.05;
    end
    
    if ~exist('ALPHA', 'var')
      ALPHA = 0.05;
    end
    numPerms = size(p_perm, 1);

    % get all pre-cluster thresholds in the orignal sample
    [clusLabel, numClus] = getPosAndNegClusters(p_sig, zscores, THRESH_SUPRACLUSTER);

    clus_Z_Stat   = zeros(1, numClus);
    clus_clusSize = zeros(1, numClus);
    % for each supra-threshold cluster, sum up the z-scores or the number
    % of pixels in this cluster
    for c = 1:numClus
        clus_Z_Stat(c)   = sum(abs(zscores(clusLabel == c))); % abs() for two-tailed testing
        clus_clusSize(c) = sum(clusLabel(:) == c);
    end

    fprintf('\nRun cluster-based multiple comparison correction... \n');

    permDist_maxSum_Z_Stat   = zeros(1, numPerms);
    permDist_maxSum_clusSize = zeros(1, numPerms);
    %% Get cluster sums (for z-scores and pixels) for all permutations
    % if you have the parallel toolbox, change this to 'parfor'
    % pool=gcp;
    % parfor (i=1:size(p_perm, 1) , pool.NumWorkers)
    for i = 1:size(p_perm, 1)
        if mod(i, 500) == 0; fprintf(['   ' num2str(i)]);  end
        [clusLabel_perm, numClus_perm] = getPosAndNegClusters(squeeze(p_perm(i,:,:)), squeeze(zscores_perm(i,:,:)), THRESH_SUPRACLUSTER);
        
        permClus_clusSize = zeros(1, numClus_perm);
        permClus_Z_Stat   = zeros(1, numClus_perm);
        % get the summed cluster stats
        for c = 1:numClus_perm
            permClus_Z_Stat(c)   = sum(abs(zscores_perm(i, clusLabel_perm == c))); % abs() for two-tailed testing
            permClus_clusSize(c) = sum(clusLabel_perm(:) == c);
        end
        
         % store only the sum of the largest cluster (based on z-scores or size) for this iteration
        if numClus_perm>0 % if significant clusters were present, otherwise leave the field at 0
             permDist_maxSum_Z_Stat(i)   = max(permClus_Z_Stat);  % minimum because z-values are all negative as norminv from p
             permDist_maxSum_clusSize(i) = max(permClus_clusSize);
        end
    end
    
    fprintf('\n');
    % now compare the obtained clusters to the permutation clusters
    clusPval_Z_Stat   = nan(numClus,1); 
    clusPval_clusSize = nan(numClus,1);
    
    % For each original supra-threshold cluster check how many of the
    % maximum sums of the permutation distribution exceeds the sum of
    % z-scores/pixels of the current cluster
    for c = 1:numClus
        valExtreme         = sum(permDist_maxSum_Z_Stat >= clus_Z_Stat(c));
        clusPval_Z_Stat(c) = valExtreme / numPerms;  
        
        valExtreme           = sum(permDist_maxSum_clusSize >= clus_clusSize(c));
        clusPval_clusSize(c) = valExtreme / numPerms;
    end
    
    % Tag all the significant clusters derived from the "sum of z-scores" 
    % method as 1 in a boolean array
    % clusPos_Z_Stat = (FREQ x TIME) or (1 x TIME/FREQ) in 1-d case
    clusPos_Z_Stat = ismember(clusLabel, find(clusPval_Z_Stat < ALPHA));   
     
    % Tag all the significant clusters derived from the "sum of pixels" 
    % (i.e. cluster size) method as 1 in a boolean array
    % clusPos_Z_Stat = (FREQ x TIME) or (1 x TIME/FREQ) in 1-d case   
    clusPos_clusSize = ismember(clusLabel, find(clusPval_clusSize < ALPHA));   

    clusPval_Z_Stat % print the p-values to the console
%     clusPval_clusSize
end


function [clusLabel, numClus] = getPosAndNegClusters(p_sig, zscores, THRESH_SUPRACLUSTER)
    % GETPOSANDNEGCLUSTERS Find significant clusters but separate them into
    %                      positive and negative clusters
    % INPUTS
    % psig         - significance matrix (FREQ x TIME)
    % zscores      - zscores to divide into positive and negative sig. clusters
    % THRESH_SUPRACLUSTER - significance threshold to determine supra-threshold 
    %                          clusters (default 0.05)
    %
    % OUTPUTS
    % clusLabel - Cluster labels (each cluster is labeled as int number
    %             starting at 1)
    % numClus   - Number of clusters

    % threshold the data
    p_subThreshold = p_sig < THRESH_SUPRACLUSTER;
    zscores_thresh = zscores.*p_subThreshold;
    % find the negative and positive clusters separately
    [clusNegative, numClus1] = bwlabeln(zscores_thresh < 0);
    [clusPositive, numClus2] = bwlabeln(zscores_thresh > 0);
    clus_tmp = clusPositive + numClus1; % increase labels by the num of neg clusters
    clus_tmp(clusPositive == 0) = 0;   % and make sure to set old 0s back to 0s
    clusLabel = clusNegative + clus_tmp;
    numClus = numClus1 + numClus2;
end