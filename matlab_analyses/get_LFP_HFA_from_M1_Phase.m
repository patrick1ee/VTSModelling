function [smoothHFA, smoothHFA_perm, p_perm_HFA_modul, avg_M1_BetaScaled, avg_STN_BetaScaled, phaseDiff_mean, phaseDiff_CI] = get_LFP_HFA_from_M1_Phase(channel, fileName, period, compute_EMG, COMPUTE_M1_HFA, evt, chanSel, wvData, subj, amplitudeThreshold, betaBand, const, paths)

NUM_PERM = 2;


% Get beta_phase diff in all channels        

if strcmp(fileName(1:9), 'JR_STN_18') && strcmp(channel, 'iM1cLFP')   
    % No ipsilateral M1 in JR_STN_18
    phaseDiff.(channel) = nan(size(wvData.C3avgEEG));
end


% Get the phase from the STN
if strcmp(const.contralat_brain.(subj), 'L')
    if numel(fileName)==13  % This is where all files are already merged
        if strcmp(channel(4:7), 'cLFP') 
            LFP_chan       = 'bipolarsLe';
        elseif strcmp(channel(4:7), 'iLFP') 
            LFP_chan       = 'bipolarsRi';
        end
    else    
        if strcmp(channel(4:7), 'cLFP') 
            LFP_chan       = chanSel.bipolarsLe;
        elseif strcmp(channel(4:7), 'iLFP') 
            LFP_chan       = chanSel.bipolarsRi;
        end
    end
    contraM1_chan  = 'C3avgEEG';
    ipsiM1_chan    = 'C4avgEEG';

elseif strcmp(const.contralat_brain.(subj), 'R')
    if numel(fileName)==13 % This is where all files are already merged
        if strcmp(channel(4:7), 'cLFP') 
            LFP_chan       = 'bipolarsRi';
        elseif strcmp(channel(4:7), 'iLFP') 
            LFP_chan       = 'bipolarsLe';
        end          
    else
        if strcmp(channel(4:7), 'cLFP') 
            LFP_chan       = chanSel.bipolarsRi;
        elseif strcmp(channel(4:7), 'iLFP') 
            LFP_chan       = chanSel.bipolarsLe;
        end     
    end
    contraM1_chan  = 'C4avgEEG';
    ipsiM1_chan    = 'C3avgEEG';
end

FLIPPED     = false;
ONLY_HI_AMP = true;


SR = wvData.sampleRate;

if strcmp(channel(1:3), 'cM1') 
    currM1_data  = wvData.(contraM1_chan);
    currOutlier2 = wvData.([contraM1_chan, '_outlier']);
elseif strcmp(channel(1:3), 'iM1')   
    currM1_data  = wvData.(ipsiM1_chan);                   
    currOutlier2 = wvData.([ipsiM1_chan, '_outlier']);
end

if strcmp(fileName(1:9), 'JR_STN_18') && strcmp(channel(1:3), 'iM1')   
    currM1_data  = nan(size(wvData.(contraM1_chan)));                   
end


if FLIPPED
%     M1_beta_flt   = ft_preproc_highpassfilter(-currM1_data, SR, betaBand(1), 4, 'but', 'twopass'); % twopass
%     addit_fl = '_flipped';
else
    M1_beta_flt = ft_preproc_highpassfilter(currM1_data, SR, betaBand(1), 4, 'but', 'twopass'); % twopass
    addit_fl = '';
end

M1_beta_flt = ft_preproc_lowpassfilter(M1_beta_flt,       SR, betaBand(2), 4, 'but', 'twopass'); % twopass
beta_amp    = abs(hilbert(M1_beta_flt));


M1_phase  = angle(hilbert(M1_beta_flt));

% figure; 
% plot(M1_phase(14000:14100)); hold on
% plot(M1_beta_flt(14000:14100)); hold on
% legend({'Beta filterd signal', 'Beta phase'})


currOutlier1 = (wvData.([LFP_chan, '_outlier']) + currOutlier2) >= 1;
M1_phase(currOutlier1) = nan;

tagAdjustments = false(size(currOutlier1));

nanEvts = isnan(evt.startChange_time);
evt.startChange_time = evt.startChange_time(~nanEvts);
evt.endChange_time   = evt.endChange_time(~nanEvts);

% if REST_ONLY
%     evt.startChange_time = false(size(tagAdjustments));
%     evt.endChange_time   = false(size(tagAdjustments));
% end
if isempty(evt.startChange_time)
    evt.startChange_time(1) = 1/SR;
    evt.endChange_time(1) = numel(tagAdjustments)/SR;
end


if  strcmp(period, 'wholeTask')
    idcs = round(evt.startChange_time(1)*SR) : round(evt.endChange_time(end)*SR);
    tagAdjustments(idcs) = true;
    
elseif strcmp(period(1:5), 'burst')
    idcs = round(evt.startChange_time(1)*SR) : round(evt.endChange_time(end)*SR);
    tagTaskPeriod(idcs) = true;
    beta_amp(~tagTaskPeriod) = nan; %without this line it takes the 50th
%     percentile from the overall recording 
    hi_beta_amp = beta_amp > prctile(beta_amp, 50);
    [clus, numClus1] = bwlabeln(hi_beta_amp);
%     figure;
    [clusLength] = hist(clus, numClus1+1);
    clusLargerThan100 = clusLength(2:end) > 100;
    allClus = 1:numClus1;
    clusOfInterest = allClus(clusLargerThan100);
    for i = clusOfInterest
        currBurst_idcs = find(clus==i);
        burst_third = round(numel(currBurst_idcs)/3);
        if strcmp(period, 'burstStart')
            idcs_burstPeriod = round(currBurst_idcs(1) : currBurst_idcs(burst_third));
            tagAdjustments(idcs_burstPeriod) = true;
        elseif strcmp(period, 'burstMiddle')
            idcs_burstPeriod = round(currBurst_idcs(burst_third+1) : currBurst_idcs(burst_third*2));
            tagAdjustments(idcs_burstPeriod) = true;
        elseif strcmp(period, 'burstEnd')
            idcs_burstPeriod = round(currBurst_idcs(burst_third*2 +1) : currBurst_idcs(end));
            tagAdjustments(idcs_burstPeriod) = true;
        end
    end
    
else
    for i = 1:numel(evt.startChange_time)
        if strcmp(period, 'adjustment')
            idcs = round(evt.startChange_time(i)*SR) : round(evt.endChange_time(i)*SR);
        elseif strcmp(period, 'frcIncrease')
            if evt.frc_increase(i)
                idcs = round(evt.startChange_time(i)*SR) : round(evt.endChange_time(i)*SR);
            else
               continue 
            end
        elseif strcmp(period, 'frcDecrease')
            if ~evt.frc_increase(i)
                idcs = round(evt.startChange_time(i)*SR) : round(evt.endChange_time(i)*SR);
            else
               continue 
            end
        elseif strcmp(period, 'earlyStable')
            idcs = round(evt.endChange_time(i)*SR) : round((evt.endChange_time(i)+2)*SR);
        elseif strcmp(period, 'stable')
            if i < numel(evt.startChange_time)
                idcs = round(evt.endChange_time(i)*SR) : round((evt.startChange_time(i+1))*SR);
            end
        elseif strcmp(period, 'firstHalf')
             if i < numel(evt.startChange_time)
                duration_toNextEvent = evt.startChange_time(i+1) - evt.startChange_time(i); 
                idcs = round(evt.startChange_time(i)*SR) : round((evt.startChange_time(i) + duration_toNextEvent/2 )*SR);  
            end
        elseif strcmp(period, 'secondHalf')
             if i < numel(evt.startChange_time)
                idcs = round((evt.startChange_time(i) + duration_toNextEvent/2 )*SR) : evt.startChange_time(i+1)*SR;
             end
        end
        tagAdjustments(idcs) = true;                    
    end
end
%     idcs = round(evt.endChange_time(i)*SR) : round(evt.startChange_time(i+1)*SR);    


if isempty(evt.startChange_time)
    tagAdjustments = true;
end

% %% ONLY LOOK AT TASK-RELATED BETA
M1_phase(~tagAdjustments) = nan;



if ONLY_HI_AMP
    if ~strcmp(period(1:5), 'burst')
        beta_amp(~tagAdjustments) = nan; %without this line it takes the 50th
    %     percentile from the overall recording 
        hi_beta_amp = beta_amp > prctile(beta_amp, amplitudeThreshold);
        M1_phase(~hi_beta_amp) = nan;    
    end
    addit_hiAmp = sprintf('_exceed%sPrctileAmp', amplitudeThreshold);
end


wvData_LFP = wvData.(LFP_chan);
if compute_EMG   
    wvData_LFP = wvData.Ext;    
%     wvData_LFP = wvData.Flx; 
%     if isempty(wvData.Flx)
%         wvData_LFP = nan(size(wvData.Ext));
%     
elseif COMPUTE_M1_HFA
    wvData_LFP = currM1_data;    
end



%% STN high-frequency activity
HFA_hi = ft_preproc_highpassfilter(wvData_LFP , SR, 150, 4, 'but', 'twopass'); % twopass
HFA    = ft_preproc_lowpassfilter(abs(HFA_hi),  SR, 100, 4, 'but', 'twopass'); % twopass


STN_beta_flt = ft_preproc_highpassfilter(wvData_LFP , SR, betaBand(1), 4, 'but', 'twopass'); % twopass
STN_beta_flt = ft_preproc_lowpassfilter(STN_beta_flt, SR, betaBand(2), 4, 'but', 'twopass'); % twopass

STN_phase  = angle(hilbert(STN_beta_flt));
STN_beta_amp  = abs(hilbert(STN_beta_flt));


%% Z-score HFA
currOutlier_STN = wvData.([LFP_chan, '_outlier'])==1;
HFA(currOutlier_STN) = nan;

% HFA    = (HFA - nanmean(HFA)) / nanstd(HFA);  % z-scored signal


% Check if STN beta needs to be flipped
do_flip = do_phase_flip(STN_beta_flt(~currOutlier_STN), STN_beta_amp(~currOutlier_STN), HFA(~currOutlier_STN), fileName, paths);

if do_flip
    STN_beta_flt = ft_preproc_highpassfilter(-wvData.(LFP_chan), SR, betaBand(1), 4, 'but', 'twopass'); % twopass
    STN_beta_flt = ft_preproc_lowpassfilter(STN_beta_flt,        SR, betaBand(2), 4, 'but', 'twopass'); % twopass
end




clear avgHFA avgBeta_M1 avgBeta_STN avgHFA_perm

dt_phases = 0.05;
phases    = -pi:dt_phases:pi;

excl = isnan(M1_phase);

valid_M1_phase     = M1_phase(~excl);
valid_HFA          = HFA(~excl);
valid_M1_beta_flt  = M1_beta_flt(~excl);
valid_STN_beta_flt = STN_beta_flt(~excl);

if strcmp(fileName(1:9), 'JR_STN_18') && strcmp(channel(1:3), 'iM1')   
    valid_M1_phase     = M1_phase;
    valid_HFA          = HFA;
    valid_M1_beta_flt  = M1_beta_flt;
    valid_STN_beta_flt = STN_beta_flt;
end


perm_HFA = [];
for perm = 1:NUM_PERM
    rng(perm)
    randNr = rand;
    randNr_constrained = randNr*0.8 + 0.1; % make sure that the cut happens not at the very beginning or end (first 10% or final 10%) 
    randIdx = ceil(randNr_constrained * numel(valid_HFA));
    perm_HFA(perm, :) = [valid_HFA(randIdx:end), valid_HFA(1:randIdx-1)];
end


for phs = 1:numel(phases)
    idcs = valid_M1_phase > phases(phs) & valid_M1_phase < phases(phs)+dt_phases;
    avgHFA(phs)        = median(valid_HFA(idcs));
    avgHFA_perm(phs,:) = median(perm_HFA(:,idcs),2);
    avgBeta_M1(phs)    = median(valid_M1_beta_flt(idcs));
    avgBeta_STN(phs)   = median(valid_STN_beta_flt(idcs));
end

% Phase diff
STN_phase = angle(hilbert(STN_beta_flt));
% M1 phase already has nans in all areas that are not meant to be included
phaseDiffs = M1_phase - STN_phase;
phaseDiffs = phaseDiffs(~isnan(phaseDiffs));
% [pval, z] = circ_rtest(phaseDiffs);
[phaseDiff_mean ul ll] = circ_mean(phaseDiffs');
phaseDiff_CI = [ul, ll];
% a = compass(cos(phaseDiffs),sin(phaseDiffs)); hold on



mirroredHFA = [fliplr(avgHFA), avgHFA, fliplr(avgHFA)];
SMOOTHING_FACTOR = 15;
smoothHFA = smooth(mirroredHFA, SMOOTHING_FACTOR);
smoothHFA = smoothHFA(numel(avgHFA)+1: numel(avgHFA)+numel(avgHFA));

smoothHFA_perm = [];
for perm = 1:NUM_PERM   
    currHFA = avgHFA_perm(:,perm);
    mirroredHFA = [fliplr(currHFA), currHFA, fliplr(currHFA)];
    curr_smoothHFA_perm = smooth(mirroredHFA, SMOOTHING_FACTOR);    
    smoothHFA_perm(:,perm) = curr_smoothHFA_perm(numel(avgHFA)+1: numel(avgHFA)+numel(avgHFA));
end

real_modul = range(smoothHFA); 
bs_modul = range(smoothHFA_perm);
% p_twotailed = (sum(abs(bs_modul-mean(bs_modul)) >= abs(real_modul-mean(bs_modul)))+1) / (NUM_PERM+1); %V2 from paper by Ernst, Permutation Methods: A Basis for Exact Inference, 2004
% p_onetailed = (sum(bs_modul >= max([real_modul, -real_modul]))+1) / (NUM_PERM+1);
p_perm_HFA_modul = (sum(bs_modul >= real_modul)+1) / (NUM_PERM+1);



% SCale beta filtered signals to the range of the HFA
avg_M1_BetaScaled  = (avgBeta_M1 - min(avgBeta_M1)) / range(avgBeta_M1);
% avg_M1_BetaScaled  = ((avgBeta_M1 - min(avgBeta_M1)) / range(avgBeta_M1) * range(smoothHFA)) + nanmean(smoothHFA) - range(smoothHFA)/2;

avg_STN_BetaScaled = ((avgBeta_STN - min(avgBeta_STN)) / range(avgBeta_STN) * range(smoothHFA)) + nanmean(smoothHFA) - range(smoothHFA)/2;


PRINT_PLOTS = false;
if PRINT_PLOTS
    fig_h = figure;
    plot(smoothHFA, 'Color', 'k', 'LineWidth', 2); hold on
    plot(avg_M1_BetaScaled, 'Color', 'r', 'LineWidth', 2)
    set(gca, 'FontSize', 18)
    set(gca, 'YTickLabel', {})   
    ylabel('a.u.')
    xlabel('Beta phase [rad]')
    xlim([1, numel(avgBeta_M1)])
    xTicks = [1, numel(avgBeta_M1)/2, numel(avgBeta_M1)];
    set(gca, 'XTick', xTicks)    
    set(gca, 'XTickLabel', {'-pi', '0', 'pi'})
    title(strrep(['M1-STN ', fileName, addit_fl, addit_hiAmp], '_', ' '))
    mkdir([paths.SaveDir, 'HFA_M1_phase_mvmt_', addit_fl, addit_hiAmp, '/'])
    export_graph(fig_h, [paths.SaveDir, '/HFA_M1_phase_mvmt_', addit_fl, addit_hiAmp, '/HFA', addit_fl, addit_hiAmp, '_',  fileName]);   
    close all
end


