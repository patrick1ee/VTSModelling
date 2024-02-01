function [] = test_HFA_coupling_sigDiff_v1_v2(Paths, Const, REF_TYPE, LOCAL_HFA)

%% Clear the workspace
% close all

HI_PASS_FREQ = 150;  % in Hz
AMPL_PRCTILE = 50;  % from 50

dt_phases = 0.1;
phases    = -pi:dt_phases:pi;



%% P1% 
% WIN_START = 5; %sec
% WIN_END   = 55; %sec

% cond2 = {  '-20',  '-65',     '25',       '70',      '115',    '160',     '205',    '250'};

%% P2 - Petra
cond1 = {'0',  '45', '90',  '135',  '180',  '225',  '270',  '315'};



WIN_START = 5; %sec
WIN_END   = 75; %sec

Z_SCORE_NORM = 1;
MIN_MAX_NORM = 0; % 1 = MIN_NORM only, 2 = MAX_NORM


%% =======

% [0:45:315]
%% ================================
SR = 1000; % recording sampling rate in Hz, do not change this

Paths.data = Paths.data;

EEG_chan = Const.EEG_chan; 
REF_CHAN = Const.REF_CHAN;  

% Select part of the data, you may want to discard some data at the start
% or end


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
    
    fig_h = figure;
    addit = ['_', num2str(HI_PASS_FREQ), 'Hz_prctile=',  num2str(AMPL_PRCTILE)];
    supertitle(strrep([currSubj, ' v1 (b) vs. v2 (r) Alpha=', REF_TYPE, ' HFA=', LOCAL_HFA, addit], '_', ' '))
    for c = 1:numel(cond1)
        
        data_tag = []; 
        conc_alpha_phase     = [];
        conc_HFA_cut         = [];
        conc_flt_data_cut    = [];
        conc_incl_high_alpha = [];
        for v = 1:2
            if v == 1
                vers = 'v1';
            elseif v == 2
                vers = 'v2';
            end

            [fileNames_tmp, shortNames, flt_band] = extract_fnames_flt_band(Paths, Const, currSubj, vers);

            %% ==========================================
            %% Prepare the data before creating individual subplots
            currFileName = fileNames_tmp{strcmp(cond1(c), shortNames)==1};
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
            % deviation) and replacing and interpolating all values that fall beyond
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


            cond_len = numel(alpha_phase);

            data_tag = [data_tag; ones(size(alpha_phase)) * v];
            conc_alpha_phase     = [conc_alpha_phase; alpha_phase];
            conc_incl_high_alpha = [conc_incl_high_alpha; incl_high_alpha]; 
            conc_HFA_cut         = [conc_HFA_cut; HFA_cut];
            conc_flt_data_cut    = [conc_flt_data_cut; flt_data_cut];
        end
        
        clear avgHFA avgAlpha 
        NUM_PERM = 100;
        for phs = 1:numel(phases)
            idcs = conc_alpha_phase > phases(phs) & conc_alpha_phase < (phases(phs)+dt_phases) & conc_incl_high_alpha;

            avgHFA(phs, 1)      = trimmean(conc_HFA_cut(idcs & data_tag == 1), 10);
%             avgAlpha(phs, 1)    = trimmean(conc_flt_data_cut(idcs, 1), 10);
            avgHFA(phs, 2)      = trimmean(conc_HFA_cut(idcs & data_tag == 2), 10);
%             avgAlpha(phs, 2)    = trimmean(conc_flt_data_cut(idcs, 2), 10);

            for perm = 1:NUM_PERM
                
                idxCut1 = randi(cond_len);
                idxCut2 = randi(cond_len);
                conc_HFA_perm = conc_HFA_cut([idxCut1:cond_len, 1:(idxCut1-1)]);

                idcs_cond2 = cond_len + [idxCut2:cond_len, 1:(idxCut2-1)];
                conc_HFA_perm = [conc_HFA_perm; conc_HFA_cut([idcs_cond2])];


                firstPart = false(size(conc_HFA_perm));
                firstPart(1:cond_len) = true;
                secondPart = false(size(conc_HFA_perm));
                secondPart(cond_len+1:end) = true;

                avgHFA_perm(phs, 1, perm) = trimmean(conc_HFA_perm(firstPart & idcs), 10);
                avgHFA_perm(phs, 2, perm) = trimmean(conc_HFA_perm(secondPart & idcs), 10);
            end
        end   


        if SMOOTH_ON
            for part = 1:2
                SMOOTHING_FACTOR = 4;

                mirroredHFA = [fliplr(avgHFA(:,part)'), avgHFA(:,part)', fliplr(avgHFA(:,part)')];
                mirroredHFA_perm = [flipud(squeeze(avgHFA_perm(:,part,:))'), ...
                                           squeeze(avgHFA_perm(:,part,:))', ... 
                                    flipud(squeeze(avgHFA_perm(:,part,:))')]';

                smoothHFA = smooth(mirroredHFA, SMOOTHING_FACTOR);
                smoothHFA_perm = [];
                for perm = 1:NUM_PERM
                    smoothHFA_perm(:,perm) = smooth(mirroredHFA_perm(:,perm), SMOOTHING_FACTOR);
                end
                avgHFA(:,part) = smoothHFA(numel(phases)+1: numel(phases)*2);
                avgHFA_perm(:,part,:) = smoothHFA_perm(numel(phases)+1: numel(phases)*2, :);
            end
        end        

        
        [coef, pval] = corr(avgHFA(:,1), avgHFA(:,2));
        
        min_norm_addit = '';
        if MIN_MAX_NORM == 1
            avgHFA_minNorm(:,1) =  avgHFA(:,1) - min(avgHFA(:,1));
            avgHFA_minNorm(:,2) =  avgHFA(:,2) - min(avgHFA(:,2));
            avgHFA_perm_minNorm(:,1,:) = avgHFA_perm(:,1,:) - repmat(min(avgHFA_perm(:,1,:)), size(avgHFA,1), 1);
            avgHFA_perm_minNorm(:,2,:) = avgHFA_perm(:,2,:) - repmat(min(avgHFA_perm(:,2,:)), size(avgHFA,1), 1);
            
            avgHFA = avgHFA_minNorm; 
            avgHFA_perm = avgHFA_perm_minNorm;
            min_norm_addit = '_MIN_NORM';
        end
        

        for perm_test = 1:3
            
            if perm_test == 1
                avgHFA_diff = avgHFA(:,1) ;
                avgHFA_perm_diff = squeeze( avgHFA_perm(:,1,:));
            elseif perm_test == 2
                avgHFA_diff = avgHFA(:,2) ;
                avgHFA_perm_diff = squeeze( avgHFA_perm(:,2,:));
            elseif perm_test == 3  % compute the difference
                avgHFA_diff = avgHFA(:,1) - avgHFA(:,2);
                avgHFA_perm_diff = squeeze( avgHFA_perm(:,1,:) - avgHFA_perm(:,2,:));
            end

            %% SIG TESTING
            rep_avgHFA_diff = repmat(avgHFA_diff, 1, NUM_PERM );
            % p_avgHFA_perm_diff_modul = (sum(avgHFA_perm_diff >= rep_avgHFA_diff)+1) / (NUM_PERM+1)
            p_twotailed = (sum(abs(avgHFA_perm_diff - repmat(mean(avgHFA_perm_diff,2), 1, NUM_PERM)) >= abs(rep_avgHFA_diff - repmat(mean(avgHFA_perm_diff,2), 1, NUM_PERM)), 2)+1) / (NUM_PERM+1); %V2 from paper by Ernst, Permutation Methods: A Basis for Exact Inference, 2004
    %         perm_z_score = (avgHFA_diff - mean(avgHFA_perm_diff,2)) ./ std(avgHFA_perm_diff,[],2); %V2 from paper by Ernst, Permutation Methods: A Basis for Exact Inference, 2004 
    %         sigPart2 = p_twotailed < 0.05;

            if Z_SCORE_NORM
                avgHFA_norm(:,perm_test) = (avgHFA_diff - mean(avgHFA_perm_diff,2)) ./ std(avgHFA_perm_diff,[],2);               
                min_norm_addit = '_Z_NORM';
            end
    
            sigPart(perm_test,:) = permstats_permsReady(avgHFA_diff, avgHFA_perm_diff');  % multiple-comparison corrected p-values
        end

        if Z_SCORE_NORM
            avgHFA(:,1:2) = avgHFA_norm(:,1:2);
        end

        subplot(3,3,c)
        p1 = plot(phases, avgHFA(:,1), 'LineWidth', 2); hold on
        p2 = plot(phases, avgHFA(:,2), 'LineWidth', 2); 

        %% Plot significance of individual HFA
        plot(phases(sigPart(1,:)), avgHFA(sigPart(1,:),1), 'x', 'LineWidth', 1, 'Color', 'r');
        plot(phases(sigPart(2,:)), avgHFA(sigPart(2,:),2), 'x', 'LineWidth', 1, 'Color', 'r');


        %% Plot significant difference between two conditions
        [clusPos,numCls] = bwlabel(sigPart(3,:));
        for cl = 1:numCls
             col = [1, 0, 0];
             if sum(avgHFA(clusPos==cl, 1)) > sum(avgHFA(clusPos==cl, 2))
                jbfill(phases(clusPos==cl), avgHFA(clusPos==cl, 1)', avgHFA(clusPos==cl, 2)', col, col); hold on
             else
                jbfill(phases(clusPos==cl), avgHFA(clusPos==cl, 2)', avgHFA(clusPos==cl, 1)', col, col); hold on              
             end                
        end         


        yLims = get(gca, 'YLim');
%             yLims = [-5 5];  % or fixed y-lims          
        % Tag the stim duration of the second condition
        stimLine_y = yLims(2) * [1 1];
        yLims(2) = yLims(2) + diff(yLims)*0.05;
        if stimStart_meanPhase<stimEnd_meanPhase  
            plot([stimStart_meanPhase, stimEnd_meanPhase], stimLine_y, 'Color', [1 0.8 0], 'LineWidth', 4);
        else
            plot([stimStart_meanPhase, phases(end)], stimLine_y, 'Color', [1 0.8 0], 'LineWidth', 4);
            plot([phases(1), stimEnd_meanPhase], stimLine_y, 'Color', [1 0.8 0], 'LineWidth', 4);
        end

        ylim(yLims)

        if c>6
            xlabel('Mu phase [rad]')
            set(gca, 'XTickLabel', {'-pi', '0', 'pi'})
        else
            set(gca, 'XTickLabel','')
        end
        xlim([phases(1), phases(end)])

        xTicks = [phases(1), 0, phases(end)];
        if c==1 || c==4 || c==7
            ylabel('HFA')
        end
        set(gca, 'XTick', xTicks) 
        set(gca, 'XTickLabel', {'-pi', '0', 'pi'}) 
        xlabel('Phase [rad]')
%             legend([p1 p2], cond1{c}, cond2{c})


        set(gca, 'FontSize', 9)
        title([cond1{c}])

    end
%     legend(h, {'v1', 'v2'}, 'Position', [0.65, 0.03, 0.2, 0.3])

    if ~exist([Paths.plots, 'HFA_diff_v1_v2/', currSubj, '/'])
        mkdir([Paths.plots, 'HFA_diff_v1_v2/', currSubj, '/'])
    end
    saveas(fig_h, [Paths.plots, 'HFA_diff_v1_v2/', currSubj, '/HFA_diff_v1v2_', '_', currSubj, '_', LOCAL_HFA, addit, min_norm_addit, '.jpg'])

end

