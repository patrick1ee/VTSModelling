function [stim_ON_win_samples, stim_OFF_win_samples] = get_stim_ON_and_OFF_windows(data, SR)

trigChan = 57;
stimPts = find(data(trigChan,:));

dataLen = numel(data(trigChan,:));

stim_ON_win_samples = [stimPts(1), stimPts(end)];
stim_OFF_win_samples = [stimPts(end), dataLen];

