function [Const] = storeConst()


% Select the frequencies you want to show for the power spectrum
Const.POW_SPECTRUM_FREQS = 5:55; % in Hz

% Select the frequency range for normalizing the power spectra
%the average of power across all conditions and all frequencies within 
% this band is used to normalize the power
Const.NORM_FREQ          = [5,40];  % in Hz, 


Const.chan_order = readtable('EEG_channel_order.csv');
Const.RE_REF_ON  = true; % this should generally be activated


% Define the colors used for creating the plots
Const.blueCol    = [14, 120, 158] / 255;
Const.darkRedCol = [169, 41, 72] / 255;
Const.yellow     = [0.8, 0.8, 0];
Const.black      = [0.3, 0.3, 0.3];


% Const.participants = {'P1'};
% Const.phases = [-65, -20, 25, 70, 115, 160, 205, 250]; % P1, Theo

Const.participants = {'P2'}; % P2, Petra, Eyes Closed
Const.phases = [0:45:315];
