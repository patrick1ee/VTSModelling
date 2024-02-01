function [] = plot_data_checks(Paths, Const)

%% Clear the workspace
close all

%% ================================
SR = 1000; % recording sampling rate in Hz, do not change this

data_path = Paths.data;

EEG_chan = Const.EEG_chan; 
REF_CHAN = Const.REF_CHAN;  


% Select part of the data, you may want to discard some data at the start
% or end
WIN_START = 5; %sec
WIN_END   = 57; %sec


%% ================================
% SOME KEY PARAMETERS THAT CAN BE TWEAKED
% FLT_DIRECTION = 'onepass'; % 'onepass' or 'twopass'; onepass is used during the recording
FLT_DIRECTION = 'twopass'; % 'onepass' or 'twopass'; onepass is used during the recording

% Define the colours for the plots
cols = [Const.blueCol; Const.darkRedCol; Const.yellow; Const.black; [1, 0, 0]; [0, 1, 0]];
brewerMap =  brewermap(11,"YlOrRd");
cols = brewerMap(3:end,:);

%% ==========================================
% Do not change these parameters
FLT_ORDER = 2;

%% ==========================================


%% Prepare the data before creating individual subplots
raw_data_multi = [];
flt_data_multi = [];
flt_data_noReref_multi = [];

for p = 1:numel(Const.participants)
    currSubj = Const.participants{p};
    
    for v = 1 %:2
        if v == 1
            vers = 'v1';
        elseif v == 2
            vers = 'v2';
        end

        [fileNames, shortNames, flt_band] = extract_fnames_flt_band(Paths, Const, currSubj, vers);


        for fn = 1:numel(fileNames)
            currFileName = fileNames{fn};
            data = h5read([Paths.data, currSubj, '/', currFileName], '/EEG');
            EEG_noReRef  =  data(EEG_chan,:)';
            
            ref_chans_avg = mean([data(REF_CHAN,:)], 1)';
        
            if Const.RE_REF_ON
                EEG = EEG_noReRef - ref_chans_avg;
            else
                EEG = EEG_noReRef;        
            end
            EEG(1) = EEG(2); % replace the first 0 with the second value
            EEG_noReRef(1) = EEG_noReRef(2);
            
        %     figure;
        %     plot(EEG)
            
            %% Read the EEG data
            if ~isnan(WIN_END)
                EEG_preSubtract = EEG(WIN_START * SR: WIN_END*SR); % Cut the data to a chosen window 
                EEG_noReRef_preSubtract = EEG_noReRef(WIN_START * SR: WIN_END*SR); % Cut the data to a chosen window 
            elseif isnan(WIN_END)
                EEG_preSubtract = EEG(WIN_START * SR: end); % Cut the data to a chosen window 
                EEG_noReRef_preSubtract = EEG_noReRef(WIN_START * SR: end); % Cut the data to a chosen window 
            end
            movingMean = movmean(EEG_preSubtract, SR*1);       % Calculate a moving average in a 1s long window...
            EEG_data = EEG_preSubtract - movingMean;          % ... and subtract it
        
            movingMean2 = movmean(EEG_noReRef_preSubtract, SR*1);       % Calculate a moving average in a 1s long window...
            EEG_noReRef_data = EEG_noReRef_preSubtract - movingMean2;          % ... and subtract it
        
            
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
        
            EEG_noReRef_data(abs(med_abs_dev_scores) > OUTL_THRESHOLD) = nan;
            EEG_noReRef_data = inpaint_nans(EEG_noReRef_data);
            
        
            %% Filter the data
            FLT_DIRECTION  ='twopass';
            [data_flt] =  butterworth_filter(data_outlierRem,  flt_band, FLT_ORDER, FLT_DIRECTION, SR);
            [data_noReRef_flt] = butterworth_filter(EEG_noReRef_data,  flt_band, FLT_ORDER, FLT_DIRECTION, SR);
            
            raw_data_multi(fn,:)  = data_outlierRem;
            flt_data_multi(fn,:)  = data_flt;
            flt_data_noReref_multi(fn,:) = data_noReRef_flt;
            
            %% Read the stimulation points and process them so they 
            % align with the selected EEG window 
            [stimPts, stimPts_affDelay] = get_stim_pts(data, WIN_START, WIN_END, SR);
            stimPts_multi.(['Rec', num2str(fn)]) = stimPts_affDelay;  % store the points for plotting
        
            alpha_phase   = angle(hilbert(data_flt));                
            stimPt_phase.(['Rec', num2str(fn)])  = alpha_phase(round(stimPts*SR));
        end
        
        
        
        %% Plot RAW DATA (check for big artefacts)
        fig_h     = figure;
        supertitle(strrep([currSubj, ' ', vers], '_', ' '))
        for fn = 1:numel(fileNames) 
            subplot(3,3,fn)
            plot(raw_data_multi(fn,:), 'Color', cols(fn,:)); hold on
            stimPts = stimPts_multi.(['Rec', num2str(fn)]);
            stimOnset = round(stimPts*SR);
            plot(stimOnset, flt_data_multi(fn, stimOnset), 'x', 'Color','k')
            title(['Raw data ', shortNames{fn}])
            xlim([1, size(raw_data_multi,2)])
            xTicks = get(gca, 'XTick');
            set(gca, 'XTick', xTicks)
            set(gca, 'XTickLabel', xTicks/SR)
            xlabel('Time [s]')
        end
        
        
        %% Plot FILTERED DATA and stim points
        fig_h     = figure;
        supertitle(strrep([currSubj, ' ', vers], '_', ' '))
        for fn = 1:numel(fileNames) 
            subplot(3,3,fn)
        %     plot(flt_data_noReref_multi(fn,:), 'Color', 0.5*[1 1 1]); hold on
            plot(flt_data_multi(fn,:), 'Color', cols(fn,:)); hold on
        
            stimPts = stimPts_multi.(['Rec', num2str(fn)]);
            stimOnset = round(stimPts*SR);
            plot(stimOnset, flt_data_multi(fn,stimOnset), 'x', 'Color','k')
            title(['Filtered data ', shortNames{fn}])
            xlim([1, size(raw_data_multi,2)])
            xTicks = get(gca, 'XTick');
            set(gca, 'XTick', xTicks)
            set(gca, 'XTickLabel', xTicks/SR)
            xlabel('Time [s]')
        end
        
        %% Plot arrows for accuracy
        fig_h     = figure;
        supertitle(strrep([currSubj, ' ', vers], '_', ' '))
        darkBlue  = [0,10,160] / 255;        
        for fn = 1:numel(fileNames) 
            subplot(3,3,fn)
        %     plot(flt_data_noReref_multi(fn,:), 'Color', 0.5*[1 1 1]); hold on
            phases = stimPt_phase.(['Rec', num2str(fn)]);
            radii = ones(size(phases));
            [u1,v1] = pol2cart(phases,radii); 
            a = compass(u1,v1); hold on
            set(gca, 'XLim', [-1 1])
            transparency = 0.1;
            set(a, 'LineWidth', 1.2, 'Color', [darkBlue, transparency]);   
            meanCart_compass(1) = mean(u1);
            meanCart_compass(2) = mean(v1);
            h = compass(meanCart_compass(1), meanCart_compass(2));
        
            [theta] = cart2pol(meanCart_compass(1), meanCart_compass(2)); 
            mean_phase = rad2deg(theta);
            % save([Paths.matfiles, 'phase_', fileNames{fn}, '.mat'], 'mean_phase')
        
        %     tweakCompass(gcf, 14)
            title(shortNames{fn})
            set(h, 'LineWidth', 2, 'Color', 'k');
        end
    end
end