function [] = plot_results_HDF5(Paths, Const, REF_CHAN)

%% Clear the workspace
close all

%% ================================
SR = 1000; % recording sampling rate in Hz, do not change this

data_path = Paths.data;


%% ================================
% SOME KEY PARAMETERS THAT CAN BE TWEAKED
% Define the size of the time-window for 
% computing the ERP (event-related potential)
ERP_tWidth = 1; % [sec]
ERP_tOffset = ERP_tWidth / 2;  % [sec]
ERP_LOWPASS_FILTER = 30; % set to nan if you want to deactivate it

% FLT_DIRECTION = 'onepass'; % 'onepass' or 'twopass'; onepass is used during the recording
FLT_DIRECTION = 'twopass'; % 'onepass' or 'twopass'; onepass is used during the recording

% Select the frequencies you want to show for the power spectrum
POW_SPECTRUM_FREQS = 6:55; % in Hz

% Define the colours for the plots
cols = [Const.blueCol; Const.darkRedCol; Const.yellow; Const.black; [1, 0, 0]; [0, 1, 0]];
brewerMap =  brewermap(8,"Set2");
cols = [0, 0, 0; ...
        brewerMap];

%% ==========================================
% Do not change these parameters
FLT_ORDER = 2;


%% Prepare the data before creating individual subplots
raw_data_multi = [];
flt_data_multi = [];
flt_data_noReref_multi = [];

for p = 1:numel(Const.participants)
    currSubj = Const.participants{p};
    
    if strcmp(currSubj, 'P2')
        % Select part of the data, you may want to discard some data at the start
        % or end
        WIN_START = 5; %sec
        WIN_END   = 75; %sec
    elseif strcmp(currSubj, 'P1')
        WIN_START = 3; %sec
        WIN_END   = 58; %sec        
    end


    
    for v = 1:2
        if v == 1
            vers = 'v1';
        elseif v == 2
            vers = 'v2';
        end

        [fileNames, shortNames, flt_band] = extract_fnames_flt_band(Paths, Const, currSubj, vers);

        for fn = 1:numel(fileNames)
            currFileName = fileNames{fn};
            data = h5read([Paths.data, currSubj, '/', currFileName], '/EEG');
                    
            if Const.RE_REF_ON                
                [Const_REF_CHAN] = return_ref_chan(REF_CHAN, Const);
                EEG_chan = Const_REF_CHAN.EEG_chan; 
                REF_chan = Const_REF_CHAN.REF_CHAN;  
                EEG  =  data(EEG_chan, :)' - mean([data(REF_chan, :)], 1)';                
            else
                EEG_noReRef  =  data(EEG_chan,:)';
                EEG = EEG_noReRef;        
            end
            
            EEG(1) = EEG(2); % replace the first 0 with the second value
%             EEG_noReRef(1) = EEG_noReRef(2);
            
            %% Read the EEG data
            if ~isnan(WIN_END)
                EEG_preSubtract = EEG(WIN_START * SR: WIN_END*SR); % Cut the data to a chosen window 
%                 EEG_noReRef_preSubtract = EEG_noReRef(WIN_START * SR: WIN_END*SR); % Cut the data to a chosen window 
            elseif isnan(WIN_END)
                EEG_preSubtract = EEG(WIN_START * SR: end); % Cut the data to a chosen window 
%                 EEG_noReRef_preSubtract = EEG_noReRef(WIN_START * SR: end); % Cut the data to a chosen window 
            end
            movingMean = movmean(EEG_preSubtract, SR*1);       % Calculate a moving average in a 1s long window...
            EEG_data = EEG_preSubtract - movingMean;          % ... and subtract it
        
%             movingMean2 = movmean(EEG_noReRef_preSubtract, SR*1);       % Calculate a moving average in a 1s long window...
%             EEG_noReRef_data = EEG_noReRef_preSubtract - movingMean2;          % ... and subtract it
        
            
            % Automatically remove artefacts occurring as sharp peaks by computing
            % a robust z-scores (based on the median and the median absolute
            % devition) and replacing and interpolating all values that fall beyond
            % 4
            med_abs_dev = 1.4826 * median(abs(EEG_data - median(EEG_data, 'omitnan')), 'omitnan');  %  the constant 1.4826 assumes normality of the data
            med_abs_dev_scores = (EEG_data - median(EEG_data, 'omitnan')) / med_abs_dev;    
            OUTL_THRESHOLD = 5;
            fprintf('%i samples removed as outlier.\n', sum(abs(med_abs_dev_scores > OUTL_THRESHOLD)));
            
        %     figure; 
        %     plot(EEG_data); hold on
        %     plot(find(abs(med_abs_dev_scores) > OUTL_THRESHOLD), EEG_data(abs(med_abs_dev_scores) > OUTL_THRESHOLD), 'x', 'Color', 'r')
            data_outlierRem = EEG_data;
            data_outlierRem(abs(med_abs_dev_scores) > OUTL_THRESHOLD) = nan;
            data_outlierRem = inpaint_nans(data_outlierRem);
        
%             EEG_noReRef_data(abs(med_abs_dev_scores) > OUTL_THRESHOLD) = nan;
%             EEG_noReRef_data = inpaint_nans(EEG_noReRef_data);
            
        
            %% Filter the data
            FLT_DIRECTION  ='twopass';
            [data_flt] =  butterworth_filter(data_outlierRem,  flt_band, FLT_ORDER, FLT_DIRECTION, SR);
%             [data_noReRef_flt] = butterworth_filter(EEG_noReRef_data,  flt_band, FLT_ORDER, FLT_DIRECTION, SR);
            
            raw_data_multi(fn,:)  = data_outlierRem;
            flt_data_multi(fn,:)  = data_flt;
%             flt_data_noReref_multi(fn,:) = data_noReRef_flt;
            
            %% Read the stimulation points and process them so they 
            % align with the selected EEG window 
            [stimPts, stimPts_affDelay] = get_stim_pts(data, WIN_START, WIN_END, SR);
            stimPts(stimPts==0) = [];
            stimPts_affDelay(stimPts_affDelay==0) = [];
            stimPts_multi.(['Rec', num2str(fn)]) = stimPts_affDelay;  % store the points for plotting
        
            alpha_phase   = angle(hilbert(data_flt));                
            stimPt_phase.(['Rec', num2str(fn)])  = alpha_phase(round(stimPts*SR));
        end
        
        
        
        %% ===================================================================
        % Calculate and plot the stimulation-evoked potential 
        % either on on 1-30 Hz filtered, or
        % the alpha/mu filtered data
        % !! Set to false if you want to look at less filtered ERPs
        ALPHA_ERP = true;

        fig_h = figure;
        supertitle(strrep([currSubj, ' ' REF_CHAN], '_', ' '))
        for subplot_row = 1:2
            subplot(3,2,(subplot_row-1)*2+1)
            if subplot_row == 1 
                itFiles = 1:5;   % iterate through the following files
            elseif subplot_row == 2
                itFiles = [1, 6:9];
            end

            for fn = itFiles
                stimPts   = stimPts_multi.(['Rec', num2str(fn)]);
                stimOnset = round(stimPts*SR);
                
                % Remove events that are too close to the data border
                events_sec = stimOnset/SR; % convert events to seconds
                dataLen_sec = size(raw_data_multi,2)/SR;  % calc. data length in seconds
                events_sec(events_sec > dataLen_sec-(ERP_tWidth-ERP_tOffset)) = nan;
                events_sec(events_sec < ERP_tOffset) = nan;
                
                EEG_data = raw_data_multi(fn,:); % unfiltered data
            
                if ~isnan(ERP_LOWPASS_FILTER)
                    [b,a]  = butter(FLT_ORDER, ERP_LOWPASS_FILTER/(SR/2), 'low');
                    EEG_data = filtfilt(b, a, EEG_data);
                end
            
                %% Activate if you want to calculate ERP on the alpha-filtered data
                if ALPHA_ERP
                    EEG_data = flt_data_multi(fn,:);
                end
            
                events_sec = events_sec(~isnan(events_sec));
                if ~isempty(events_sec)
                    events_sec(1) = [];
                end
                trialsMatrix = getTrialsMatrix(events_sec, EEG_data, 1/SR, ERP_tWidth, ERP_tOffset);
            
                plot(median(trialsMatrix,2), ':', 'LineWidth', 1.5, 'Color', cols(fn,:)); hold on
                xTicks = [ 250, 500, 750]; % here you define the position of the x-ticks
                set(gca, 'XTick', xTicks)
                xlim([1, size(trialsMatrix,1)])
                set(gca, 'XTickLabel', ((xTicks/SR*1000 - ERP_tOffset*1000)))
            %     xlim([1, size(trialsMatrix,1)])
                xlabel('Time [ms]')
                
                title(['Event-rel. potent. ', vers])
            end
            yLims = get(gca, 'YLim');
            plot(ERP_tOffset*SR * [1 1], yLims, '--', 'Color', 'k', 'LineWidth', 2)
        end
        
        
        %% ===================================================================
        % Calculate and plot the power spectra
        winLength = SR;  
        win = hamming(winLength);
        overlap = 0.5; % with 50 percent overlap    
       

         for subplot_row = 1:2
            subplot(3, 2, (subplot_row-1)*2+2) % add subplots to position 2 and 4
            if subplot_row == 1 
                itFiles = 1:5;   % iterate through the following files
            elseif subplot_row == 2
                itFiles = [1,6:9];
            end
        
            for fn = itFiles
                % Details on the pwelch function: https://edoras.sdsu.edu/doc/matlab/toolbox/signal/pwelch.html
                powSpect = pwelch(raw_data_multi(fn,:)',  win, winLength * overlap, 
                , SR, 'psd');    
                h(fn) = plot(powSpect, 'LineWidth', 1, 'Color', cols(fn,:)); hold on
                if fn == 1  % make the black noStim condition line thicker, as it is in the background
                    h(fn) = plot(powSpect, 'LineWidth', 1.5, 'Color', cols(fn,:)); hold on
                end
            %     h(fn) = plot(smooth(powSpect, 5), 'LineWidth', 2, 'Color', cols(fn,:)); hold on
                xTicks = 1:5:numel(POW_SPECTRUM_FREQS);
                set(gca, 'XTick', xTicks);
                freqCenter = POW_SPECTRUM_FREQS(xTicks);
                xlim([1, numel(powSpect)])
                set(gca, 'XTickLabel', freqCenter);            
                [~, findFreq] = min(abs(POW_SPECTRUM_FREQS - mean(flt_band)));
                yLims = get(gca, 'YLim');
                plot(findFreq*[1,1], yLims, ':', 'Color', 'k', 'LineWidth', 1)
            %     set(gca, 'YLim', yLims)
                title(['Power spectra ', vers] )
                xlabel('Frequencies [Hz]')
            end
            if subplot_row == 2
%                 subplot(3, 2, 5)
%                 plot([0,0], '.')
                legend(h, strrep(shortNames, '_', ' '), 'Position', [0.65, 0.03, 0.2, 0.3])
            end
         end
         if ~exist([Paths.plots, 'powSpectra/', currSubj, '/'])
            mkdir([Paths.plots, 'powSpectra/', currSubj, '/'])
         end
         print(fig_h, '-dpng', [Paths.plots, 'powSpectra/', currSubj, '/powSpectra_', currSubj, '_', REF_CHAN, '_', vers, '_', '.png'], '-r300')       
    end
end
