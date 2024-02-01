function [] = plot_HFA_coupling_localHFA(Paths, Const, REF_TYPE, LOCAL_HFA)

%% Clear the workspace
% close all

HI_PASS_FREQ = 150;  % in Hz
AMPL_PRCTILE = 50;

dt_phases = 0.1;
phases    = -pi:dt_phases:pi;


%% ================================
SR = 1000; % recording sampling rate in Hz, do not change this

Paths.data = Paths.data;

EEG_chan = Const.EEG_chan; 
REF_CHAN = Const.REF_CHAN;  

% Select part of the data, you may want to discard some data at the start
% or end
WIN_START = 5; %sec
WIN_END   = 75; %sec


%% ================================
% SOME KEY PARAMETERS THAT CAN BE TWEAKED

% FLT_DIRECTION = 'onepass'; % 'onepass' or 'twopass'; onepass is used during the recording
FLT_DIRECTION = 'twopass'; % 'onepass' or 'twopass'; onepass is used during the recording

SMOOTH_ON = true;

%% ==========================================
% Do not change these parameters
FLT_ORDER = 2;

RE_REF_ON = true;


for p = 1:numel(Const.participants)
    currSubj = Const.participants{p};
    
    for v = 1:2
        if v == 1
            vers = 'v1';
        elseif v == 2
            vers = 'v2';
        end

        [fileNames, shortNames, flt_band] = extract_fnames_flt_band(Paths, Const, currSubj, vers);

        fig_h = figure;
        addit = ['_', num2str(HI_PASS_FREQ), 'Hz_prctile=',  num2str(AMPL_PRCTILE)];
        supertitle(strrep([currSubj, ' ', vers, ' Alpha=', REF_TYPE, ' HFA=', LOCAL_HFA, addit], '_', ' '))
        %% ==========================================
        %% Prepare the data before creating individual subplots
        for fn = 1:numel(fileNames)
            currFileName = fileNames{fn};
            data = h5read([Paths.data, currSubj, '/', currFileName], '/EEG');
            EEG_noReRef  =  data(EEG_chan, :)';
            
            % C3 = 15, POz = 29, Pz = 31
            % 9 = FC5, 10 = FC1, 20 = CP5, 21 = CP1    
            ref_chans_avg = mean([data(REF_CHAN, :)], 1)';
        
            if RE_REF_ON
                EEG = EEG_noReRef - ref_chans_avg;
            else
                EEG = EEG_noReRef;        
            end
            EEG(1) = EEG(2); % replace the first 0 with the second value
            EEG_noReRef(1) = EEG_noReRef(2);
        
            [Const_localHFA]  = return_ref_chan(LOCAL_HFA, Const);
            EEG_chan_localHFA = Const_localHFA.EEG_chan; 
            REF_CHAN_localHFA = Const_localHFA.REF_CHAN;  
            EEG_localHFA  =  data(EEG_chan_localHFA, :)' - mean([data(REF_CHAN_localHFA, :)], 1)';
        
            
            %plot(EEG)
            
            %% Read the EEG data
            if ~isnan(WIN_END)
        %         try
                EEG_preSubtract = EEG(WIN_START * SR: WIN_END*SR); % Cut the data to a chosen window 
                EEG_localHFA_pS = EEG_localHFA(WIN_START * SR: WIN_END*SR); 
        %         catch
        %             EEG_preSubtract = EEG(WIN_START * SR: end); % Cut the data to a chosen window 
        %             EEG_localHFA_pS = EEG_localHFA(WIN_START * SR: end); 
        %             
        %         end
            else
                EEG_preSubtract = EEG(WIN_START * SR: end); % Cut the data to a chosen window 
                EEG_localHFA_pS = EEG_localHFA(WIN_START * SR: end); 
            end
            movingMean = movmean(EEG_preSubtract, SR*2);      % Calculate a moving average in a 2s long window...
            EEG_data   = EEG_preSubtract - movingMean;          % ... and subtract it
%             figure;
%             plot(EEG_preSubtract); hold on
%             plot(movingMean, 'LineWidth', 2); 
%             figure;
%             plot(EEG_data)
           
            movingMean_localHFA = movmean(EEG_localHFA_pS, SR*2); % Calculate a moving average in a 2s long window...
            EEG_localHFA        = EEG_localHFA_pS - movingMean_localHFA;
            
            % Automatically remove artefacts occurring as sharp peaks by computing
            % a robust z-scores (based on the median and the median absolute
            % devition) and replacing and interpolating all values that fall beyond
            % 4
            med_abs_dev = 1.4826 * median(abs(EEG_data - median(EEG_data, 'omitnan')), 'omitnan');  %  the constant 1.4826 assumes normality of the data
            med_abs_dev_scores = (EEG_data - median(EEG_data, 'omitnan')) / med_abs_dev;    
            OUTL_THRESHOLD = 5;
            numel(EEG_data(med_abs_dev_scores > OUTL_THRESHOLD))
        %     figure; 
        %     plot(EEG_data); hold on
        %     plot(find(abs(med_abs_dev_scores) > OUTL_THRESHOLD), EEG_data(abs(med_abs_dev_scores) > OUTL_THRESHOLD), 'x', 'Color', 'r')
            data_outlierRem = EEG_data;
            data_outlierRem(abs(med_abs_dev_scores) > OUTL_THRESHOLD) = nan;
            data_outlierRem = inpaint_nans(data_outlierRem);
            
            %% Mirror the EEG data chunk and append it at the end and at the beginning, 
            % which avoids sharp edges on each side and improves the quality of the filtering
            orig_Length = numel(data_outlierRem);
        
            % Butterworth filter on re-ref data
            flt_data_cut = butterworth_filter(data_outlierRem, flt_band, FLT_ORDER, FLT_DIRECTION, SR);
        
            % Filtering to get HFA
            EEG_localHFA   = [flipud(EEG_localHFA); EEG_localHFA; flipud(EEG_localHFA)];
            [b,a] = butter(FLT_ORDER, HI_PASS_FREQ/(SR/2), 'high');
            HFA_hi = filtfilt(b, a, EEG_localHFA);
            [b,a] = butter(FLT_ORDER, 50/(SR/2), 'low');
            HFA   = filtfilt(b, a, abs(HFA_hi));
            HFA_cut = HFA(orig_Length+1:orig_Length*2);
        
            % Extract the phase and amplitude of alpha/mu activity
            alpha_phase = angle(hilbert(flt_data_cut));
            alpha_ampl  = abs(hilbert(flt_data_cut));
            incl_high_alpha = alpha_ampl > prctile(alpha_ampl, AMPL_PRCTILE);
        
            %% Get  stim points to get average  phase during stim
            stimDuration  = 0.04;
            [stimPts, stimPts_affDelay] = get_stim_pts(data, WIN_START, WIN_END, SR);
        
            stimStart_points_samples = round((stimPts_affDelay)*SR);
            stimEnd_points_samples   = round((stimPts_affDelay  + stimDuration)*SR);
        
            newWinLen = (WIN_END-WIN_START)*SR; % need to make sure that when adding the delays, 
            % we remove all stim points that exceed the window length, which would throw an error
            stimStart_points_samples(stimStart_points_samples > newWinLen) = [];
            stimEnd_points_samples(stimEnd_points_samples     > newWinLen) = [];
        
            stimStart_phases = alpha_phase(stimStart_points_samples);
            stimEnd_phases   = alpha_phase(stimEnd_points_samples);
        
        
            %% Compute  circular  mean, see also: https://github.com/circstat/circstat-matlab/blob/master/circ_mean.m
            % https://insidebigdata.com/2021/02/12/circular-statistics-in-python-an-intuitive-intro/ 
            % The following lines are complex notation, using Euler's formula, but they are equivalent to:
            % [u1,v1] = pol2cart(stimStart_phases, ones(size(stimStart_phases))); 
            % [theta] = cart2pol(mean(u1), mean(v1)); 
            stimStart_meanPhase = angle(sum(exp(1i*stimStart_phases)));
            stimEnd_meanPhase   = angle(sum(exp(1i*stimEnd_phases)));
        
            clear avgHFA avgAlpha 
            NUM_PERM = 100;
            
            for phs = 1:numel(phases)
                idcs = alpha_phase > phases(phs) & alpha_phase < (phases(phs)+dt_phases) & incl_high_alpha;
                avgHFA(phs)      = trimmean(HFA_cut(idcs), 10);
                avgAlpha(phs)    = trimmean(flt_data_cut(idcs), 10);
        
                for perm = 1:NUM_PERM
                    constrainRand = rand*0.6+0.2;
                    numDataPoints = numel(HFA_cut);
                    cutPoint = round(numDataPoints*constrainRand);
                    HFA_cut_perm = [HFA_cut(cutPoint:end); HFA_cut(1:cutPoint-1)];
                    avgHFA_perm(phs,perm)  = trimmean(HFA_cut_perm(idcs), 10);
        %             avgHFA_perm(phs,perm)  = median(HFA_cut_perm(idcs));
                end
            end   
        
            avgAlpha_scaled  = avgAlpha / max(avgAlpha);
        %     if fn == 1
               meanHFA =  mean(avgHFA);
        %     end
            avgHFA_centered  = avgHFA - meanHFA;
            avgHFA_perm_centered      = avgHFA_perm - repmat(mean(avgHFA_perm,1), size(avgHFA,1), 1);
        %     avgHFA_perm_centered      = avgHFA_perm - repmat(meanHFA, size(avgHFA,1), 1);
        
        %     figure; plot(mirroredHFA_perm(:,1)); hold on
        %     plot(64:(64+62),avgHFA_perm_centered(:,1), 'Color', 'r')
            
            avgHFA_normed = avgHFA_centered';
        
            avgHFA_perm_normed = avgHFA_perm_centered;
        
            if SMOOTH_ON
                SMOOTHING_FACTOR = 4;
                
                mirroredHFA = [fliplr(avgHFA_centered), avgHFA_centered, fliplr(avgHFA_centered)];
                mirroredHFA_perm = [flipud(avgHFA_perm_centered)', avgHFA_perm_centered', flipud(avgHFA_perm_centered)']';
        
                smoothHFA = smooth(mirroredHFA, SMOOTHING_FACTOR);
                smoothHFA_perm = [];
                for perm = 1:NUM_PERM
                    smoothHFA_perm(:,perm) = smooth(mirroredHFA_perm(:,perm), SMOOTHING_FACTOR);
                end
                smoothHFA = smoothHFA(numel(avgHFA)+1: numel(avgHFA)+numel(avgHFA));
                smoothHFA_perm = smoothHFA_perm(numel(avgHFA)+1: numel(avgHFA)+numel(avgHFA), :);
                
                avgHFA_normed = smoothHFA;
                avgHFA_perm_normed = smoothHFA_perm;
            end
        
        
            %% SIG TESTING
            rep_avgHFA_normed = repmat(avgHFA_normed, 1, NUM_PERM );
            % p_avgHFA_perm_normed_modul = (sum(avgHFA_perm_normed >= rep_avgHFA_normed)+1) / (NUM_PERM+1)
            p_twotailed = (sum(abs(avgHFA_perm_normed - repmat(mean(avgHFA_perm_normed,2), 1, NUM_PERM)) >= abs(rep_avgHFA_normed - repmat(mean(avgHFA_perm_normed,2), 1, NUM_PERM)), 2)+1) / (NUM_PERM+1); %V2 from paper by Ernst, Permutation Methods: A Basis for Exact Inference, 2004
            perm_z_score = (avgHFA_normed - mean(avgHFA_perm_normed,2)) ./ std(avgHFA_perm_normed,[],2); %V2 from paper by Ernst, Permutation Methods: A Basis for Exact Inference, 2004
            subplot(3,3,fn)
        
            maxVal  = max(avgHFA_normed);
            avgHFA_normed  = avgHFA_normed / maxVal; %maxVal;
        
            
            sigPart = p_twotailed < 0.05;
            %% Version 1: normalized to 1
        %     plot(avgHFA_normed, 'Color', 'k', 'LineWidth', 2); hold on
        %     plot(find(sigPart), avgHFA_normed(sigPart), 'x', 'Color', 'r', 'LineWidth', 2); hold on
        %     plot(avgAlpha_scaled, 'Color', 'r', 'LineWidth', 2)
            


            %% Version 2: Permstats t-scores 
            % plot(phases, avgAlpha_scaled * max(perm_z_score), 'Color', 0.7*[1,1,1], 'LineWidth', 0.5); hold on
            plot(phases, avgAlpha_scaled * 4, 'Color', 0.7*[1,1,1], 'LineWidth', 0.5); hold on
            plot(phases(sigPart==1), perm_z_score(sigPart), 'o', 'Color', 'r', 'LineWidth', 2, 'MarkerSize', 3); 
            plot(phases, perm_z_score, 'Color', 'k', 'LineWidth', 1.8); 

            % either have free-floating
            yLims = get(gca, 'YLim');
            yLims = [-5 5];  % or fixed y-lims
           
            ylim(yLims)
            if fn > 1 % only plot the stimulation indicaton for the STIM ON conds
                plot(stimStart_meanPhase*[1,1], yLims, ':', 'Color', 'r', 'LineWidth', 2);
                plot(stimEnd_meanPhase*[1,1], yLims, ':', 'Color', 'k');
            end
        
            %% Plot the duration of stimulation perception 
            
            set(gca, 'FontSize', 9)
            %% Version 1
        %     set(gca, 'YTickLabel', {})   
        %     ylabel('a.u.')
            %% Version 2
            if fn==1 || fn==4 || fn==7
                ylabel('z-scores')
            end
            if fn>6
                xlabel('Mu phase [rad]')
                set(gca, 'XTickLabel', {'-pi', '0', 'pi'})
            else
                set(gca, 'XTickLabel','')
            end
            xlim([phases(1), phases(end)])
           
            xTicks = [phases(1), 0, phases(end)];
            set(gca, 'XTick', xTicks)    
            title([shortNames{fn}])
        end
        if ~exist([Paths.plots, 'localHFA/'])
            mkdir([Paths.plots, 'localHFA/'])
        end
        saveas(fig_h, [Paths.plots, 'localHFA/HFA_', currSubj, '_', vers, '_', LOCAL_HFA, addit, '.jpg'])
    end
end
close all