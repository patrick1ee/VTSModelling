function [] = plot_results_topography(Paths, Const)

%% Clear the workspace
% close all
tic

%% ================================
SR = 1000; % recording sampling rate in Hz, do not change this
NEW_SR = 200;

% Select the frequencies you want to show for the power spectrum
POW_SPECTRUM_FREQS = 4:55; % in Hz
COH_FREQS = 3:2:32;

% rec_labels = {'EO rest', 'EO trick'};
% fileNames = {'30_06_2023_P1_Ch14_FRQ=9Hz_FULL_CL_phase=90_EO_rest_noStim_v1_', ...
%              '30_06_2023_P1_Ch14_FRQ=9Hz_FULL_CL_phase=90_EO_trick_noStim_v1_'};
% 
% rec_labels = {'EO rest', 'EO trick'};
% fileNames = {'30_06_2023_P1_Ch14_FRQ=9Hz_FULL_CL_phase=90_EO_stand_noStim_v1_', ...
%              '30_06_2023_P1_Ch14_FRQ=9Hz_FULL_CL_phase=90_EO_stand_trick_noStim_v1_' }


rec_labels = Const.dataLabels;
fileNames = Const.fileNames;


TARGET_FREQ = mean(Const.FLT_BAND); % Hz

data_path = Paths.data;

C3 = 15;
Cz = 16;
C4 = 17;
CP1 = 21; %sensory neck left
CP2 = 22; %sensory neck right
Pz = 26;
P3 = 25;
P4 = 27;
T7 = 14;
T8 = 18;
O1 = 30;
O2 = 32;
Oz = 31;
F3 = 5;
F4 = 7;
FC1 = 10;
FC2 = 11;

EMG1 = 33;
EMG2 = 34;
EMG3 = 35;
EMG4 = 36;

%% ====================================================================
% CHANGE SETTINGS HERE
EEG_chans = {'C3', 'CP1', 'CP2', 'P3', 'P4'}; 
% EEG_chans = {'T7', 'T8', 'O1', 'O2', 'Cz', 'Pz'}; 
EEG_chans = {'C3', 'C4', 'F3', 'F4', 'FC1', 'FC2'}; 
EEG_chans = {'C3'}; 
EMG_chan  = 'EMG2'; 
EMG_chan  = 'C4'; 


RE_REF_ON = Const.RE_REF_ON; 
REF_CHAN  = Const.REF_CHAN;  
% REF_CHAN  = [1:32];   % AVERAGE REFERENCE

CUT_DATA_TO_WIN = true;  % necessary to match the file lengths exactly
WIN_START = 0.01; %sec
WIN_END   = 110; %sec

%% ====================================================================



% Define the colours for the plots
Const.blueCol    = [14, 120, 158] / 255;
Const.darkRedCol = [169, 41, 72] / 255;
Const.orange = [255, 192, 0] / 255;
Const.black  = [0.3, 0.3, 0.3];
Const.lightBlue  = [136, 196, 228] /255;
cols = [Const.blueCol; Const.darkRedCol; Const.black;  [1, 0, 0];  Const.lightBlue; Const.orange];


%% ==========================================
% Do not change these parameters
FLT_ORDER = 2;

%% ==========================================

%% Prepare the data before creating individual subplots
raw_data = [];
EMG_data = [];
for fn = 1:numel(fileNames)
    currFileName = fileNames{fn};
    data = h5read([data_path, currFileName, '.hdf5'], '/EEG');

    for chan = 1:numel(EEG_chans)

        chan_nr = eval(EEG_chans{chan});
        EEG_noReRef  =  data(chan_nr,:)';
        
        % C3 = 15, POz = 29, Pz = 31
        % 9 = FC5, 10 = FC1, 20 = CP5, 21 = CP1    
        ref_chans_avg = mean([data(REF_CHAN,:)], 1)';
        EMG = data(eval(EMG_chan), :)';
   
        if RE_REF_ON
            EEG = EEG_noReRef - ref_chans_avg;
            if ~strcmp(EMG_chan(1:2), 'EM')
                EMG = EMG - ref_chans_avg;
            end
        else
            EEG = EEG_noReRef;
        end
    
        EEG(1) = EEG(2); % replace the first 0 with the second value
        EEG_noReRef(1) = EEG_noReRef(2);
                
        %% Read the EEG data
        if CUT_DATA_TO_WIN
            EEG_preSubtract = EEG(WIN_START * SR: WIN_END * SR); % Cut the data to a chosen window 
            EEG_noReRef_preSubtract = EEG_noReRef(WIN_START * SR: WIN_END*SR); % Cut the data to a chosen window 
            EMG = EMG(WIN_START * SR: WIN_END*SR);
        else
            EEG_preSubtract = EEG; % Cut the data to a chosen window 
            EEG_noReRef_preSubtract = EEG_noReRef;
        end
        movingMean = movmean(EEG_preSubtract, SR*2);       % Calculate a moving average in a 2s long window...
        EEG_data = EEG_preSubtract - movingMean;          % ... and subtract it
    
        movingMean2 = movmean(EEG_noReRef_preSubtract, SR*2);       % Calculate a moving average in a 2s long window...
        EEG_noReRef_data = EEG_noReRef_preSubtract - movingMean2;          % ... and subtract it
    
        
        % Automatically remove artefacts occurring as sharp peaks by computing
        % a robust z-scores (based on the median and the median absolute
        % devition) and replacing and interpolating all values that fall beyond
        % 4
        med_abs_dev = 1.4826 * median(abs(EEG_data - median(EEG_data, 'omitnan')), 'omitnan');  %  the constant 1.4826 assumes normality of the data
        med_abs_dev_scores = (EEG_data - median(EEG_data, 'omitnan')) / med_abs_dev;    
        OUTL_THRESHOLD = 5;
        fprintf('%i samples removed as outl. in %s.\n', sum(abs(med_abs_dev_scores > OUTL_THRESHOLD)), EEG_chans{chan});
    %     figure; 
    %     plot(EEG_data); hold on
    %     plot(find(abs(med_abs_dev_scores) > OUTL_THRESHOLD), EEG_data(abs(med_abs_dev_scores) > OUTL_THRESHOLD), 'x', 'Color', 'r')
        data_outlierRem = EEG_data;
        data_outlierRem(abs(med_abs_dev_scores) > OUTL_THRESHOLD) = nan;
        EEG_outlierRem = inpaint_nans(data_outlierRem);
    
        [fsorig, fsres] = rat(SR / NEW_SR);      
        data_DS  = resample(EEG_outlierRem', fsres, fsorig)';  
    
        % Remove the appended edges again and store it:
        raw_data.(EEG_chans{chan}).(['rec', num2str(fn)]) = data_DS;

        if chan == 1
            movingMean = movmean(EMG, SR*2);       % Calculate a moving average in a 2s long window...
            EMG_subtract = EMG - movingMean;    % ... and subtract it
            FLT_ORDER = 2;
            if strcmp(EMG_chan(1:2), 'EM')
                EMG = abs(butterworth_filter(EMG_subtract,  [20, 200], FLT_ORDER, 'twopass', SR));  
            end
            EMG_DS  = resample(EMG', fsres, fsorig)';  
            EMG_data.(['rec', num2str(fn)])  = EMG_DS;
        end
    end
end





%% Calculate and plot the power spectra
winLength = NEW_SR*4;  
win = hamming(winLength);
overlap = 0.5; % with 50 percent overlap    
supertitle([strrep(fileNames{1}, '_', ' ')])
for fn = 1:numel(fileNames)+1
    for EEG_or_COH = 1:2        
        subplot(3,2,(fn-1)*2 + EEG_or_COH)
        clear leg_h
        for chan = 1:numel(EEG_chans)     

            if fn==numel(fileNames)+1 % launch the diff plot
                if EEG_or_COH == 1 
                    diff_data = diff(store_data.POW.(EEG_chans{chan}));
                    
                elseif EEG_or_COH == 2 
                    diff_data = diff(store_data.COH.(EEG_chans{chan}));
                end
            end
                if EEG_or_COH == 1
                    if fn < numel(fileNames)+1
                        EEG = raw_data.(EEG_chans{chan}).(['rec', num2str(fn)]);
                        % Details on the pwelch function: https://edoras.sdsu.edu/doc/matlab/toolbox/signal/pwelch.html
                        powSpect = pwelch(EEG',  win, winLength * overlap, POW_SPECTRUM_FREQS, NEW_SR, 'psd');    
                        to_plot = smooth(powSpect, 2);
                    else
                        to_plot = diff_data;
                    end
                    leg_h(chan) = plot(to_plot, 'LineWidth', 2, 'Color', cols(chan,:)); hold on 
                    ylabel('Power')
                    xLim2 = size(powSpect,2);
                    curr_freqs = POW_SPECTRUM_FREQS;
                    store_data.POW.(EEG_chans{chan})(fn,:) = smooth(powSpect, 2);
                elseif EEG_or_COH == 2
                    if fn < numel(fileNames)+1
                        EEG = raw_data.(EEG_chans{chan}).(['rec', num2str(fn)]);
                        EMG = EMG_data.(['rec', num2str(fn)]);
                        
                        hamm_win = hamming(1*NEW_SR);
                        overlap = numel(hamm_win)/2;
                        [ms_coh, f] = mscohere(EEG, EMG, hamm_win, overlap, COH_FREQS, NEW_SR);
                        % [ms_coh, f] = mscohere(EEG, EMG, [], [], COH_FREQS,NEW_SR);
                        % Pxy         = cpsd(EEG, EMG, window, overlap, COH_FREQS, NEW_SR);
                        % phase_diffs = -angle(Pxy)/pi*180;

                        
                        for fi = 1:numel(COH_FREQS)         

                            sig1_full = butterworth_filter(EEG,  [COH_FREQS(fi)-1, COH_FREQS(fi)+1], FLT_ORDER, 'twopass', NEW_SR);
                            sig2_full = butterworth_filter(EMG,  [COH_FREQS(fi)-1, COH_FREQS(fi)+1], FLT_ORDER, 'twopass', NEW_SR);

                            percentageOverlap = overlap / numel(hamm_win);
                            numWins = numel(EEG) / numel(hamm_win) / percentageOverlap;

                            spectcoher_win = [];
                            imagcoher_win = [];
                            PLV_win = [];
                            for i = 1:(numWins-1)
                                idcs_win   = 1:numel(hamm_win);
                                currentWin = ((i-1)*numel(hamm_win) * percentageOverlap) + idcs_win;

                                sig1 = hilbert(sig1_full(currentWin) .* hamm_win)';
                                sig2 = hilbert(sig2_full(currentWin) .* hamm_win)';

                                %% From Cohen Chapter 26: % compute power and cross-spectral power                            
                                % alternative notation:
                                spec1 = mean(abs(sig1).^2,2);
                                spec2 = mean(abs(sig2).^2,2);
                                specX = abs(mean( abs(sig1).*abs(sig2) .* exp(1i*(angle(sig1)-angle(sig2))) ,2)).^2;
                                % % compute spectral coherence
                                spectcoher_win(i) = specX ./ (spec1.*spec2);

                                % imaginary coherence
                                spec1 = sum(sig1.*conj(sig1),2);
                                spec2 = sum(sig2.*conj(sig2),2);
                                specX = sum(sig1.*conj(sig2),2);
                                imagcoher_win(i) = abs(imag(specX ./ sqrt(spec1 .* spec2)));

                                phaseDiffs = angle(sig1) - angle(sig2);
                                PLV_win(i) = abs(nanmean(exp(1i*phaseDiffs), 2));

                            end
                            spectcoher(fi) = mean(spectcoher_win); % average over all hamming windows
                            imagcoher(fi) = mean(imagcoher_win);

                            PLV(fi) = mean(PLV_win);
                                    
                            to_plot = spectcoher;                            
                        end
                    else
                        to_plot = diff_data;
                    end

                    leg_h(chan) = plot(to_plot, 'LineWidth', 2, 'Color', cols(chan,:)); hold on
                    plot(ms_coh, 'LineWidth', 2, 'Color', 'r'); hold on
                    ylabel('Coherence')
                    xLim2 = size(spectcoher,2);
                    curr_freqs = COH_FREQS;
                    store_data.COH.(EEG_chans{chan})(fn,:) = to_plot;
               end
            end
            xTicks = 1:5:numel(curr_freqs);
            set(gca, 'XTick', xTicks);
            freqCenter = curr_freqs(xTicks);
            xlim([1, xLim2])
            set(gca, 'XTickLabel', freqCenter);            
            [~, findFreq] = min(abs(curr_freqs - TARGET_FREQ));
            yLims = get(gca, 'YLim');
            plot(findFreq*[1,1], yLims, '--', 'Color', 'k', 'LineWidth', 2)
            % set(gca, 'YLim', yLims)
            if EEG_or_COH == 1
                addTitle = 'EEG ';
            else
                addTitle = 'COH ';
            end

            if fn == 3
                title([addTitle, 'Diff 1-2'])
                plot([1, xLim2], [0,0], ':', 'Color', 0.5*[1 1 1])
            else
                title([addTitle, rec_labels{fn}])
            end
            xlabel('Frequencies [Hz]')
            legend(leg_h, EEG_chans)
        end
end
    
toc
