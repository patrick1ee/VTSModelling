%[RCs,surfFromBin,hilbertPhaseVectDeg,Xblock,Yblock,phaseVect,deltaPhaseVect,deltaAmpVect]...
function    [RCs,surfFromBin,phaseVect,deltaPhaseVect,deltaAmpVect]...
    =blockMethod2(...
    signal,...
    stimTrig,...
    fs,...
    blockIdx,... % n x 2
    blockPhaseVect,...
    patientIdx,... % same
    binningOption,... % 0: adaptive binning, 1: use phaseCodes (provided in blockPhaseVect) and do not redo binning, 2: use phaseCodes and redo binning
    deepVerbose,...
    absChangeInAmp,...
    nPhaseBins,...
    nAmpBins,...
    avgFunc,...
    phaseDelay,...
    ampBrkdOptions)

%%% 30-11-18    cluster version
%%% FULL version living concurrently with the cluster version
%%% 19-09-18    can also return phaseVect, deltaPhaseVect and deltaAmpVect for testing experimental PRCs
%%% 25-06-18    added measure late and amp ref before block options
%%% 18-05-18    changed file name, using doPhAmpBinning to possibly output response surfaces
%%% 01-05-18    - phaseDelay should be used to introduce a correction for the phase of the stim pulses (when
%%%             binningOption~=1) -> +pi/2 for positive 0 Xing. This value should be used in the prc fitting pipeline
%%%             for synthetic data. When evaluating patient data, various values may be used to account for the variable
%%%             phase reference of phaseCodes. 
%%%				- Also plotting phase tracking is now controlled by a bool       
%%% 24-04-18	testing with synthetic data
%%%             changed median(stim pulse angles) to avgAngle (second avgAngle in code) -> impacts on fit
%%%             switch for +pi/2
%%%				changed fitting method to polyfit to center and scale for better fits
%%% 21-03-18 	making sure there is at least 3 stim pulses per block (otherwise getStartStopIdx will fail)
%%% in the past: bug correction with getStartStop
%% WORK TO BE DONE TO REUSE IN COMBINATION WITH SINGLE PULSE CODE
assert(ismember(binningOption,[0 1 2]),'binning option should be 0, 1 or 2');%throw error if false
removeOutliers=false;

plotPhaseTracking=false;

measureLater=false;
ampRefBeforeBl=false;

% pdist_t=2;
% thresh=0.8;%3.8;

time=(1:length(signal))/fs;

% pdist_n=pdist_t*fs;
% trialIdx=get_blocks(stimTrig,thresh,pdist_n);
% idxForMean=[];
% for k=1:size(trialIdx,1)
%     idxForMean=[idxForMean,trialIdx(k,1):trialIdx(k,2)];
% end
% wholeSignalEnv=abs(hilbert(signal));
% meanOverTrials=mean(wholeSignalEnv(idxForMean));
% stdOverTrials=std(wholeSignalEnv(idxForMean));


nBlocks=size(blockIdx,1);

phaseVect=zeros(nBlocks,1);
ampVect=zeros(nBlocks,1);
deltaPhaseVect=zeros(nBlocks,1);
deltaAmpVect=zeros(nBlocks,1);
blockToRemoveIdx=[];
outlierDeltaT=[];
outlierBlIdx=[];
trackedPh=cell(1,nBlocks);
% stimIdxDebug=[] %%%%%%%%%%%%
trigIdx=find(stimTrig>0); %%%%%% debug
toRmIdx=[];

%
% fullHPh=angle(hilbert(signal)); %%%%%% debug

for k=1:nBlocks
    
    %%% indexing
    blockRange=blockIdx(k,1):blockIdx(k,2);
    noStimTime=1;%%%%%%%% 1
    blockRangePlus=blockRange(1)-round(noStimTime*fs):...
        blockRange(end)+round(noStimTime*fs);
    
    %%% counting number of pulses
    %         stimPdist_n=stimPdist_t*fs;
    %         stimIdx=get_blocks(stimTrig(blockRangePlus),stimThresh,stimPdist_n);
    stimIdxCont=trigIdx(trigIdx>blockRangePlus(1) & trigIdx<blockRangePlus(end));%%%%%%%%%%%%% THIS MAY BREAK FOR OTHER APPLICATIONS
    
    % making sure there is at least 3 stim pulses per block (otherwise getStartStopIdx will fail)
    if ~isempty(stimIdxCont) && length(stimIdxCont)>3
        
        stimIdxCont=stimIdxCont-blockRangePlus(1)+1;%%%%%%%%%%%%%%
        
        [start,stop]=getStartStopIdx(stimIdxCont);%%%%%%%%%%%%%%%%%% BETTER WITH THIS - SEEMS OK WITH PATIENT DATA HAYRIYE TWO AND SYNTHETIC DATA
        stimIdx=[start,stop];%[stimIdx,stimIdx];%%%%%%%%%%%%%%%%%%
        nPulses=size(stimIdx,1);
        %%%% for debug
        %         stimIdxDebug=[stimIdxDebug;stimIdx+blockRangePlus(1)-1];
        
        signalHilbert=hilbert(signal(blockRangePlus));%%% detrend here!-> removed!
        %%% phase
        % truePhase could be removed
        truePhase=[];
        if isempty(truePhase)
            hilbertPhase=angle(signalHilbert);
        else
            hilbertPhase=truePhase{k};
        end
        
        unwPhase=unwrap(hilbertPhase);
        
        unwPhaseNoStim=unwPhase(1:round(noStimTime*fs));
        
        [pfit,~,mu]=polyfit(time(blockRangePlus(1):blockRange(1)-1)',...
            unwPhaseNoStim',1);
        
        pfitX=[time(blockRangePlus(1));time(blockRangePlus(end))];
        pfitY=polyval(pfit,pfitX,[],mu);
		
		
        
        %%% stim phase
        if binningOption==1
            phaseVect(k)=blockPhaseVect(k);%phasevect given in rd no deg!
        else
            sgPulsePhases=zeros(nPulses,1);
            for i=1:size(stimIdx,1)
                %                 sgPulsePhases(i)=median(fullHPh(stimIdx(i,1)+blockRangePlus(1)-1:stimIdx(i,2)+blockRangePlus(1)-1)+pi/2);
  
                sgPulsePhases(i)=avgAngle(hilbertPhase(max(1,stimIdx(i,1)):max(1,stimIdx(i,2)))+phaseDelay);
            end
            phaseVect(k)=avgAngle(sgPulsePhases);
            trackedPh{k}=sgPulsePhases;
        end
        
        
        %%% amplitude
        amp=abs(signalHilbert);
        
        if ampRefBeforeBl
            ampVect(k)=mean(amp(1:blockRange(1)-blockRangePlus(1))); % simplify, this is calculated soon after
        else
            ampVect(k)=mean(amp);
        end
        
        %%% removing outliers
        if removeOutliers
            
            inATrough=false;
            idxStart=length(amp);
            
            for i=1:length(amp)
                if amp(i)<meanOverTrials-stdOverTrials && inATrough==false
                    idxStart=i;
                    inATrough=true;
                end
                if amp(i)>meanOverTrials-stdOverTrials && inATrough==true
                    inATrough=false;
                    outlierBlIdx=[outlierBlIdx,k];
                    outlierDeltaT=[outlierDeltaT,(i-idxStart)/fs];
                end
                if (round((i-idxStart)/fs)>4 && inATrough==true)
                    %                     || (i==length(amp) && inATrough==true)
                    blockToRemoveIdx=[blockToRemoveIdx,k];
                    outlierBlIdx=[outlierBlIdx,k];
                    outlierDeltaT=[outlierDeltaT,(i-idxStart)/fs];
                    break
                end
            end
            
        end
        
        %%%
        %         noStimAmp=mean(amp(blockRangePlus(1):blockRange(1)-1));
        noStimAmp=mean(amp(1:blockRange(1)-blockRangePlus(1)));
        
        
        ampNoStimRange=round(noStimTime*fs);
        %         stimAmp=mean(amp(blockRange-blockRangePlus(1)+1));
        if measureLater
            mesRange=(blockRange(end)-blockRangePlus(1)+1):(-blockRangePlus(1)+blockRangePlus(end)+1);
            stimAmp=mean(amp(mesRange));
        else
            mesRange=(blockRange(end-ampNoStimRange)-blockRangePlus(1)+1):(blockRange(end)-blockRangePlus(1)+1);
            stimAmp=mean(amp(mesRange));
        end
        %%% deltas
        
        % delta phase: difference in phase at the end of the stim block between the hilbert phase and the line fitted to
        % the phase noStimTime before the block (currently taken as 1s)
        
        if measureLater
            deltaPhase=unwPhase(-blockRangePlus(1)+blockRangePlus(end)+1)...
                -polyval(pfit,time(blockRangePlus(end)),[],mu); 
        else   
            deltaPhase=unwPhase(-blockRangePlus(1)+blockRange(end)+1)...
                -polyval(pfit,time(blockRange(end)),[],mu);   
        end
        
        % delta amp: difference in mean amp from end of stim block - noStimTime and mean amp noStimTime before stim
        % block
        if absChangeInAmp
            deltaAmp=(stimAmp-noStimAmp);
        else
            deltaAmp=(stimAmp-noStimAmp)/noStimAmp;
        end
        
        if isnan(deltaAmp) || isnan(deltaPhase)
            keyboard
        end
        
        deltaPhaseVect(k)=deltaPhase/nPulses;
        deltaAmpVect(k)=deltaAmp/nPulses;
        
        
        %%% plots
        if deepVerbose
            figure
            ColOrd = get(gca,'ColorOrder');
            suptitle(['Block #' num2str(k) ...
                ' - stimulation phase = ' num2str(round(rad2deg360(phaseVect(k)))) ' deg'])
            h1=subplot(3,1,1);
            plot(time(blockRangePlus),signal(blockRangePlus),'lineWidth',2)
            hold on
            plot(time(blockRangePlus(1):blockRange(1)-1),...
                amp(1:blockRange(1)-blockRangePlus(1)),'lineStyle','--','lineWidth',2)
            plot(time(blockRange(end-ampNoStimRange):blockRange(end)),...
                amp(blockRange(end-ampNoStimRange)-blockRangePlus(1)+1:blockRange(end)-blockRangePlus(1)+1),'lineWidth',2)
            xlim([time(blockRangePlus(1)) time(blockRangePlus(end))])
            set(gca,'XTick',[]);
            ylabel('filtered signal')
%             xlabel('time (s)')
            
            h2=subplot(3,1,2);
            hold on
            plot(pfitX,pfitY,'Col',ColOrd(2,:),'lineStyle','--','displayName','reference','lineWidth',1)
            plot(time(blockRangePlus),unwPhase,'Col',ColOrd(3,:),'displayName','with stimulation','lineWidth',1)
            %         plot(time(blockRangePlus),hilbertPhase)
            xlim([time(blockRangePlus(1)) time(blockRangePlus(end))])
            ylim([min(min(pfitY),min(unwPhase)) max(max(pfitY),max(unwPhase))])
            set(gca,'XTick',[]);
            ylabel('Hilbert phase (rd)')
%             xlabel('time (s)')
            legend('location','best')
            
            h3=subplot(3,1,3);
            plot(time(blockRangePlus),stimTrig(blockRangePlus),'k','lineWidth',0.2)
            xlim([time(blockRangePlus(1)) time(blockRangePlus(end))])
            set(gca,'YTick',[]);
            ylabel('triggers')
            xlabel('time (s)')
            linkaxes([h1 h2 h3],'x')
            mySaveasFlex('fNameNoNowStr','blockMethod','fontSize',15,'size','M')
            keyboard
            
            showBlAndSig=false;
            blAndSigInONEplot=true;
            if showBlAndSig
                if ~blAndSigInONEplot
                    suptitle(['Block #' num2str(k) ...
                        ' - stimulation phase = ' num2str(round(rad2deg360(phaseVect(k)))) ' deg'])
                    h1=subplot(2,1,1);
                    plot(time(blockRangePlus),signal(blockRangePlus),'lineWidth',2)
                    xlim([time(blockRangePlus(1)) time(blockRangePlus(end))])
                    set(gca,'XTick',[]);
                    ylabel('filtered signal')

                    h2=subplot(2,1,2);
                    plot(time(blockRangePlus),stimTrig(blockRangePlus),'k','lineWidth',0.2)
                    xlim([time(blockRangePlus(1)) time(blockRangePlus(end))])
                    set(gca,'YTick',[]);
                    ylabel('triggers')
                    xlabel('time (s)')
                    linkaxes([h1 h2],'x')
                    mySaveasFlex('fNameNoNowStr','blockDesign','fontSize',15,'size','M')
                else
                    figure
                    title(['stimulation phase = ' num2str(round(rad2deg360(phaseVect(k)))) ' deg'])
                    yyaxis left
                    plot(time(blockRangePlus),signal(blockRangePlus),'lineWidth',2)
                    xlim([time(blockRangePlus(1)) time(blockRangePlus(end))])
                    ylabel('filtered tremor acceleration (a.u.)')

                    yyaxis right
                    plot(time(blockRangePlus),stimTrig(blockRangePlus),'lineWidth',0.5)
                    xlim([time(blockRangePlus(1)) time(blockRangePlus(end))])
                    set(gca,'YTick',[]);
                    ylabel('stimulation triggers (a.u.)')
                    xlabel('time (s)')
                    
                    mySaveasFlex('fNameNoNowStr','blockDesign','fontSize',19,'size','M')
                end
            end
        end
        
    else
        
        toRmIdx=[toRmIdx,k];
        
    end
    
end

if removeOutliers
    phaseVect(blockToRemoveIdx)=[];
    deltaPhaseVect(blockToRemoveIdx)=[];
    deltaAmpVect(blockToRemoveIdx)=[];
end

if ~isempty(toRmIdx)
    
    phaseVect(toRmIdx)=[];
    deltaPhaseVect(toRmIdx)=[];
    deltaAmpVect(toRmIdx)=[];
    trackedPh(toRmIdx)=[];
    blockPhaseVect(toRmIdx)=[];
    nBlocks=nBlocks-length(toRmIdx);
    
    if nBlocks < 120
        phaseBinsCenter=NaN(nPhaseBins,1);
        semDeltaPhase=NaN(nPhaseBins,1);
        meanDeltaPhase=NaN(nPhaseBins,1);
        meanDeltaAmp=NaN(nPhaseBins,1);
        semDeltaAmp=NaN(nPhaseBins,1);
        hilbertPhaseVectDeg=[];
        Xblock=[];
        Yblock=[];
        return
    end
    
end

if binningOption~=1
    
    Xhist=[];
    Yhist=[];
	
    for k=1:nBlocks
        nTrPh=length(trackedPh{k});
        Xhist=[Xhist;rad2deg360(blockPhaseVect(k))*ones(nTrPh,1)];
        Yhist=[Yhist;rad2deg360(trackedPh{k})];
    end
	
	Xblock=rad2deg360(blockPhaseVect);
    Yblock=rad2deg360(phaseVect);
    
    if plotPhaseTracking
	
	az=-64.3;
    el=71.6;
	
	    figure
        hist3([Xhist,Yhist],[24,256],'edgeColor','none')
        set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
        title(['phase tracking assessment - patient ' num2str(patientIdx)])
        xlabel('target phase (deg)')
        ylabel('actual hilbert phase (deg)')
        view([az,el])
        
        dimX=15;
        dimY=10;
        resolution='-r300';% does not change the size of plots, labels...
        changeFontBool=false;
        fontSize=[];
        changeLineThBool=false;
        lineTh=2;
        myDir='fig';
        mkdir(myDir);
        fileNameNoNowStr=[myDir '/phaseTracking'];
        mySaveas( gcf, gca, dimX, dimY, resolution,...
            changeLineThBool, lineTh,...
            changeFontBool, fontSize,...
            fileNameNoNowStr );
        
    %
    %     figure
    %     scatter(rad2deg360(blockPhaseVect),rad2deg360(phaseVect))
    %     title('phase tracking assessment')
    %     xlabel('target phase (deg)')
    %     ylabel('actual hilbert phase (deg)')
    

    
        figure
        hist3([Xblock(:),Yblock],[24,256],'edgeColor','none')
        set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
        title(['phase tracking assessment - mean by blocks - patient ' num2str(patientIdx)])
        xlabel('target phase (deg)')
        ylabel('actual hilbert phase (deg)')
        view([az,el])
        
        dimX=15;
        dimY=10;
        resolution='-r300';% does not change the size of plots, labels...
        changeFontBool=false;
        fontSize=[];
        changeLineThBool=false;
        lineTh=2;
        myDir='fig';
        mkdir(myDir);
        fileNameNoNowStr=[myDir '/phaseTracking'];
        mySaveas( gcf, gca, dimX, dimY, resolution,...
            changeLineThBool, lineTh,...
            changeFontBool, fontSize,...
            fileNameNoNowStr );
			
	end
    
end



% %% Debug: stimTrig(stimVect) vs stimIdxDebug
%
% stimIdxToComp=find(stimTrig==1);
% unique(stimIdxDebug(:,1)==stimIdxToComp)
%
%
% %% Debug: comparing Yblock and Ybl
%
% Debug=true;
% unique(Yblock==Ybl)
%
% figure
% plot(Yblock==Ybl)

%% Response curves

%% Obtaining response curves and surfaces

[~,surfFromBin] = doPhAmpBinning( patientIdx,...
                                binningOption,...
                                blockPhaseVect,...%needed if binningOption is 2 (if 1, the phase codes are in hilbertPhaseVect)
                                phaseVect,...
                                ampVect,...
                                deltaPhaseVect,...
                                deltaAmpVect,...
                                nPhaseBins,... %does not do anything if binningOption=1;
                                nAmpBins,...
                                avgFunc,... %for instance mean or median
                                1,...%numOfPulsesForNorm = 1, the bl mehtod already takes care of normalisation                                                    
                                ampBrkdOptions.checkBinningPlot,...
                                ampBrkdOptions.plotResponseCurves,...
                                ampBrkdOptions.plotOnlyExtremeBins,... %only has an effect if nRCAmpBin>1
                                ampBrkdOptions.plotSurface );

doScalingPlots=false;                            
if doScalingPlots             
% useAvgSqr=false;                            
% getRespScaleVsAmp(surfFromBin,useAvgSqr,true);
getRespScaleVsAmpStd(surfFromBin,true,patientIdx);%,nPhaseBins)
getRespScaleVsAmpProp(surfFromBin,true,patientIdx)                     
end

RCs = doPhAmpBinning   ( patientIdx,...
                                binningOption,...
                                blockPhaseVect,...%needed if binningOption is 2 (if 1, the phase codes are in hilbertPhaseVect)
                                phaseVect,...
                                ampVect,...
                                deltaPhaseVect,...
                                deltaAmpVect,...
                                nPhaseBins,... %does not do anything if binningOption=1;
                                1,... %one amp bin to get the overall response curve
                                avgFunc,... %for instance mean or median
                                1,...%numOfPulsesForNorm = 1, the bl mehtod already takes care of normalisation                                                    
                                ampBrkdOptions.checkBinningPlot,...
                                ampBrkdOptions.plotResponseCurves,...
                                false,... %only has an effect if nRCAmpBin>1
                                false);


end