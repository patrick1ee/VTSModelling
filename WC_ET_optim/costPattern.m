function cost=costWithPRC(x, dataFeatures, filename, stimInSigmoid, includeShift, nTr, df)
%%% 18-12-18	    don't attempt to write in file if filename is '' (for NaNtests)
%%% 17-12-18        added df as an input
%%% 16-12-18        try statement as some failures (integration to huge values it seems, then kdensity fails)
%%%                 cost is NaN if NaN in PRC - compatible with Patternsearch
%%% 02-12-18        writing cost5 to the disk
%%% 01-12-18        now relying on blockMethod2 for coherence with dataFeatures and reading
%%%                 coeff PSD back to default
%%%                 possibility of including the shift as a feature
%%%                 cost jump for PRC if nNaN>0
%%%                 including nTr as a param

try
    %%%% PARAMETERS
    wIE=x(1);
    wEI=x(2);
    wEE=x(3);
    beta=x(4);
    Tau=x(5);
    thetaE=x(6);
    thetaI=x(7);
    sigma=abs(x(8));
    stimMag=abs(x(9));%no negative stim for now
    delayMs=abs(x(10));
    
    dt=1E-3;
    
    
    %%%% STIM MAT
    %%%% stimProgramme mat:
    %1st column: stim amp, 0 if no stim
    %2nd column: phase between 0 and 2pi, -99 when stim no applied
    %3rd column: phase index used by Gihan's mex, not implemented
    %1 row per timestep
    
    rampUpTime=40;%in s
    % nTr=90;
    nBlperTr=12;
    interBlTime=1;%in s
    interTrTime=5;%in s
    blTime=5;%in s
    
    nBl=nBlperTr*nTr;
    
    toIdx=@(t)round(t/dt);
    rampUpN=toIdx(rampUpTime);
    interBlN=toIdx(interBlTime);
    interTrN=toIdx(interTrTime);
    blN=toIdx(blTime);
    
    N=rampUpN+(nTr+1)*interTrN+(nBl-nTr)*interBlN+nBl*blN;
    % one inter trial interval after ramp up time
    
    stim_prog=zeros(N,3);
    stim_prog(:,2)=-99*ones(N,1);
    
    phases=linspace(0,2*pi,nBlperTr+1);% so there is no stim target at 360
    phases=phases(1:end-1);% so there is no stim target at 360
    
    blockPhaseVect=[];
    blockIdx=[];
    trialIdx=zeros(nTr,2);
    
    for i=1:nTr
        
        trPhases=phases(randperm(nBlperTr));
        
        for j=1:nBlperTr
            
            idxStart=1+rampUpN+...% ramp up time
                i*interTrN+...% inter trial interval
                (i-1)*(nBlperTr*blN+(nBlperTr-1)*interBlN)+...% previous trials (complete)
                (j-1)*(blN+interBlN);% previous blocks from the current trial
            idx=idxStart:(idxStart+blN-1);
            stim_prog(idx,1)=stimMag;
            stim_prog(idx,2)=trPhases(j);
            
            blockPhaseVect=[blockPhaseVect,trPhases(j)];% for block method
            blockIdx=[blockIdx;[idx(1) idx(end)]];% for block method
            
            if j==1
                trialIdx(i,1)=idx(1);
            elseif j==nBlperTr
                trialIdx(i,2)=idx(end);
            end
            
        end
        
    end
    
    %%%% INTEGRATION WITH STIM
    
    [E,~,stimVect]=...
        doEulerNoiseTrackNStimCluster(wIE,wEI,wEE,beta,Tau,thetaE,thetaI,N,sigma,dt,stim_prog,delayMs,stimInSigmoid);
    
    Emean=mean(E(rampUpN:end));
    Estd=std(E(rampUpN:end));
    Eproc=(E-Emean)/Estd;
    %Eproc=zscore(E);% no need for zscoring trials only as no garbage between trials in model
    Eproc=detrend(Eproc);
    
    %%%% FEATURE COMPUTATION
    
    nPartialTr=9;
    partialIdx=randperm(nTr,nPartialTr);
    partialTrIdx=trialIdx(partialIdx,:);
    
    % PSD
    %%% maybe verify trial indices are right? -> done
    %%% make sure bad blocks are removed? -> not for now
    PSDverbose=false;
    yPSDmod = getTrPSD( Eproc, 1/dt, partialTrIdx, dataFeatures.PSD.xPSD, dataFeatures.PSD.wLratio, PSDverbose ) ;
    
    % change to energy later on
    [maxPSD,maxIdx]=max(yPSDmod);
    fCenter=dataFeatures.PSD.xPSD(maxIdx);
    
    % amplitude PDF
    % remove bad blocks? -> not for now
    % idx=[];
    % for k=1:nTr
    %     idx=[idx,trialIdx(k,1):trialIdx(k,2)];
    % end
    env = abs(hilbert(Eproc));
    PDFverbose=false;
    yPDFmod = getTrAmpPDF( [], env, partialTrIdx, dataFeatures.ampPDF.xPDF, PDFverbose ) ;
    
    % amplitude PSD
    % check first few and last few values: not needed, doing hilbert of the whole signal
    % remove bad blocks? -> not for now
    envPSDverbose=false;
    yEnvPSDmod = getTrAmpPSD( [], env, 1/dt, partialTrIdx, dataFeatures.envPSD.xEnvPSD, dataFeatures.envPSD.wLratio, envPSDverbose );
    
    % (block) PRC
    % fixed bins to be able to compare, but rebinning according to hilbert phase
    % actual number of pulses per block? yes
    % block phase = avg of recalculated phases? yes -> angle avg
    
    % pre-processing
    doFilteringBool=true;
    bandSpecified=true;
    fminus=max(0.01,fCenter-df);
    fplus=fCenter+df;
    aroundMean=true;
    dfminus=2;
    dfplus=2;
    filterVerbose=false;
    
    if doFilteringBool
        Eproc = filteringProcess(E,...
            1/dt,...
            bandSpecified,...
            fminus,...
            fplus,...
            aroundMean,...
            dfminus,...
            dfplus,...
            filterVerbose);
    end
    
    
    nPhaseBins=12;
    ampBrkdOptions.plotResponseCurves = false;
    ampBrkdOptions.checkBinningPlot = false;
    ampBrkdOptions.plotOnlyExtremeBins = false;
    ampBrkdOptions.plotSurface = false;
    
    [RCs]...
        =blockMethod2(...
        Eproc',...%signal
        stimVect,...%stimTrig
        1/dt,...%fs.
        blockIdx,...%blockIdx
        blockPhaseVect,...%blockPhaseVect
        '',... % patientIdx
        2,... % % 0: adaptive binning, 1: use phaseCodes (provided in blockPhaseVect) and do not redo binning, 2: use phaseCodes and redo binning
        false,...%deepVerbose
        true,... %bool, abs change in amp if true, relative if false
        nPhaseBins,...%nPhaseBins
        1,...%nAmpBins
        @mean,...%avgFunc,...
        0,...%phaseDelay -> to set for actual bl PRC data feature
        ampBrkdOptions);%ampBrkdOptions
    
    xPRCmod=RCs.pRCBinsCenterMat;
    yPRCmod=RCs.meanDeltaPhase;
    % semPRCmod=RCs.semDeltaPhase;
    yARCmod=RCs.meanDeltaAmp;
    % semARCmod=RCs.semDeltaAmp;
    
    if includeShift
        shiftMod = getRCshift(xPRCmod(:),yPRCmod(:),xPRCmod(:),yARCmod(:));
        
        xPRCdat = dataFeatures.PRC.xPRC;
        yPRCdat = dataFeatures.PRC.yPRC;
        xARCdat = dataFeatures.ARC.xARC;
        yARCdat = dataFeatures.ARC.yARC;
        shiftDat = getRCshift(xPRCdat(:),yPRCdat(:),xARCdat(:),yARCdat(:));
        cost5 = 3*abs(shiftDat-shiftMod)/pi;
    else
        cost5 = 0;
    end
    
    
    % [~,yPRCmod,~]... %x in deg, y in rd
    %     =blockMethodCluster(...
    %     Eproc',...
    %     stimVect,...
    %     1/dt,...
    %     blockIdx,...
    %     blockPhaseVect,... % between 0 - 2pi
    %     2,... % % 0: adaptive binning, 1: use phaseCodes (provided in blockPhaseVect) and do not redo binning, 2: use phaseCodes and redo binning
    %     nPhaseBins);
    
    % we define pulse here as x stim diracs (Gihan's data is structured this way)
    %nbPerPulse=1;% handled in block method
    
    nNaN=sum(isnan(yPRCmod));
    
    coeffPSD=1/4;
    coeffPDF=1/4;
    coeffenvPSD=1/4;
    coeffPRC=1/4;
    
    yPSDdat=dataFeatures.PSD.yPSD;
    yPDFdat=dataFeatures.ampPDF.yPDF;
    yEnvPSDdat=dataFeatures.envPSD.yEnvPSD;
    yPRCdat=dataFeatures.PRC.yPRC;
    
    
    % / data variance -> to check
    cost1=sum((yPSDdat-yPSDmod).^2)/sum((yPSDdat-mean(yPSDdat)).^2);
    cost2=sum((yPDFdat-yPDFmod).^2)/sum((yPDFdat-mean(yPDFdat)).^2);
    cost3=sum((yEnvPSDdat-yEnvPSDmod).^2)/sum((yEnvPSDdat-mean(yEnvPSDdat)).^2);
    if nNaN>0%nPhaseBins-round(nPhaseBins/2)
        cost4=NaN;
    else
        cost4=nansum((yPRCdat-yPRCmod).^2)/(nPhaseBins-nNaN)*nPhaseBins/sum((yPRCdat-mean(yPRCdat)).^2);
    end
    
    cost = coeffPSD*cost1+coeffPDF*cost2+coeffenvPSD*cost3+coeffPRC*cost4+cost5;
    
    % assert(~isnan(cost),'cost is NaN')
    
    if ~isempty(filename)
      fileID = fopen(filename,'a');
      fprintf(fileID,'%.10f %.10f %.10f %.10f %.13f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\r\n',...
          wIE,wEI,wEE,beta,Tau,thetaE,thetaI,sigma,stimMag,delayMs,cost,cost1,cost2,cost3,cost4,cost5,fCenter,maxPSD,Estd,nNaN);
      fclose(fileID);
    end
catch
    cost = NaN;
end

end

