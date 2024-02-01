function [] = cluster_permStats(diff, diff_perm)

%% Cluster based stats
diff_sum_perm_mean = squeeze(mean(diff_perm)); % get the mean of the permutation distribution (TIME x FREQ)
diff_sum_perm_std  = squeeze(std(diff_perm));  % get the std of the permutation distribution (TIME x FREQ)
if oneDim % if there's only a TIME or FREQ dimension:
    zscores = (diff' - diff_sum_perm_mean) ./ diff_sum_perm_std;
    diffPerm_mean = diff_sum_perm_mean;
    diffPerm_std  = diff_sum_perm_std;
else  % otherwise for TIME * FREQ
    zscores = (diff-diff_sum_perm_mean) ./ diff_sum_perm_std; % zscore the real difference relative to the mean and std of the permutation difference
    diffPerm_mean(1,:,:) = diff_sum_perm_mean;
    diffPerm_std(1,:,:)  = diff_sum_perm_std;
end
p_orig = 2 * (1 - normcdf(abs(zscores), 0, 1)); % get p-values from the zscore, abs to make it 2-tailed

% calculate: (diff_perm - diffPerm_mean) / diffPerm_std
zscores_perm = bsxfun(@rdivide, bsxfun(@minus, diff_perm, diffPerm_mean), diffPerm_std);
p_perm =  2 * (1 - normcdf(abs(zscores_perm), 0, 1)); % get p-values from the zscore, abs to make it 2-tailed

preCluster_thresh = .05;
alpha = .05;
% preCluster_thresh = .2;
% alpha = .2;
% Obtain multiple-comparison corrected p-values for each suptra-threshold
% cluster (clusPval_Z_Stat), and the position of significant clusters
% (clusPos_Z_Stat)
% clusPval_Z_Stat and clusPos_Z_Stat are based on the sum of z-scores
% clusPval_clusSize & clusPos_clusSize are based on the cluster size
% irrespective of the z-scores. In most cases the Z_Stat or clusSize
% result is identical.
[clusPval_Z_Stat, clusPos_Z_Stat, clusPval_clusSize, clusPos_clusSize] = getSignifClusters(p_orig, zscores, p_perm, zscores_perm, preCluster_thresh, alpha);
