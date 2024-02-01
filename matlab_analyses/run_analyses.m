clear all
close all

addpath(genpath('.')) % Add all subfolders in the current folder to the Matlab paths




%% Define the paths that contain the recorded data and will be used 
%  to store plots and matfiles
Paths.data     = 'data\';
Paths.data     = 'F:\Experiments\3rd_Yr_Projects\CL_stim\data\';
Paths.plots    = 'plots\'; % specify the path relative to the folder where the code is
Paths.matfiles = 'matfiles\'; % specify the path relative to the folder where the code is


%% Load constant variables (incl. subject numbers, and color definitions)
Const = storeConst();

EEG_LOC = 'C3_POz';  % EEG location, this was the tracking signal
[Const] = return_ref_chan(EEG_LOC, Const);


% step1_clean_outliers(Paths, Const)  %% Replace, make fit for current code

% plot_data_checks(Paths, Const)
% 
% localRefs = {'C3_local', 'C4_local', 'CP1_local', 'CP2_local', 'CP5_local', 'CP6_local', ...
%              'P3_local', 'P4_local', 'FC1_local', 'FC2_local', 'F3_local', 'F4_local', ...
%              'POz_local'};
localRefs = {'C3_local', 'C4_local', 'CP1_local', 'CP2_local', 'POz_local'};
for r = 1:numel(localRefs)
    plot_powSpectra_ERP(Paths, Const, localRefs{r})
    test_HFA_coupling_sigDiff_twoConds(Paths, Const, EEG_LOC, localRefs{r})
    test_HFA_coupling_sigDiff_v1_v2(Paths, Const, EEG_LOC, localRefs{r})
%     plot_HFA_coupling_localHFA(Paths, Const, EEG_LOC, localRefs{r})
end

