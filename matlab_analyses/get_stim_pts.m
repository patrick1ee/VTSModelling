function [stimPts, stimPts_affDelay] = get_stim_pts(data, WIN_START, WIN_END, SR)

trigChan = 57;

stimPts = find(data(trigChan,:)) / SR;
diff_triggers = diff(stimPts);
tagStart  = diff_triggers' > 0.02;  % look for start of each pulse, which was 4*100 Hz (i.e. 0.01 intervals), only take the first point of each pulse
stimPts = stimPts([true; tagStart]); 

if ~isnan(WIN_END)
    stimPts(stimPts<WIN_START) = []; 
    stimPts(stimPts>WIN_END) = [];    

    stimPts = stimPts - WIN_START;  % correct for the starting time 

    afferentDelay = 0.019; % https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3814810/, onset of N2 at around 16ms, peak at around 19
    %afferentDelay = 0; 
    stimPts_affDelay = stimPts + afferentDelay;
    newWin_duration = WIN_END - WIN_START;
    stimPts_affDelay(stimPts_affDelay > newWin_duration) = [];    
end


fprintf('Nr stim points = %i.\n', numel(stimPts));