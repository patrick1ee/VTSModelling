function [RCs,surfFromBin] = doPhAmpBinning( dataTag,...
    binningOption,...
    blockPhaseVect,...%needed if binningOption is 2 (if 1, the phase codes are in hilbertPhaseVect)
    hilbertPhaseVect,...
    ampAtStimVect,...
    deltaPhaseVect,...
    deltaAmpVect,...
    nPhaseBins,... %does not do anything if binningOption=1 or 2;
    nRCAmpBin,...
    avgFunc,... %for instance mean or median
    numOfPulsesForNorm,...
    checkBinningPlot,...
    plotResponseCurves,...
    plotOnlyExtremeBins,... %only has an effect if nRCAmpBin>1
    plotSurface )

%%% 30-11-18    Cluster version
%%% 28-11-18    Modifying bin centers to avoid binning point close to 360 with 0 -> set useSimpleBins = true;
%%% 20-06-18    If nPhaseBins=1 -> avgFunc and stdFunc are such that the scaling is given in the surf.
%%%             Bug patch for pBin==1: second condition has to also satisfy wrapTo360(phaseBins(pBin))>180 (otherwise
%%%             seen to fail for binning option = 0)
%%% 28-05-18    Making sure blockPhaseVect is used as a column for binningOption 2
%%%             Sorting out unequal first and last bin for binningOption 2 - does it work well with H data?
%%% 18-05-18    creating a function for this so it is reusable
%%% to do       maybe integrate the boxplot code at some point (see end of this file)

plotForGihan=false;
extremeAndMed=false;%only work if above is true and plotOnlyExtremes is false
saveForGihan=false;% plot for Gihan has to be true

useSimpleBins = false;

% converting x-axis data (stim phase) to deg
hilbertPhaseVectDeg=rad2deg360(hilbertPhaseVect);

% normalisation by number of pulses per period
deltaPhaseVect=deltaPhaseVect/numOfPulsesForNorm;
deltaAmpVect=deltaAmpVect/numOfPulsesForNorm;

% defining amplitude bins
[ampRCBins,ampRCBinsCenter]=getBins(ampAtStimVect,nRCAmpBin);
ampRCBinVect=1:length(ampRCBins)-1;

% defining phase bins for binningOptions 1 and 2
if binningOption==1
    phaseBinsCenter=unique(hilbertPhaseVectDeg);
    nPhaseBins=length(phaseBinsCenter);%shadowing the param given as input to the function in that case
elseif binningOption==2
    phaseBinsCenter=unique(rad2deg360(blockPhaseVect(:)));
end
pRCBinVect=1:nPhaseBins;

% initialising main response surfaces' cells
deltaPhaseCell=cell(nRCAmpBin,nPhaseBins);
deltaAmpCell=cell(nRCAmpBin,nPhaseBins);
pRCBinsCenterMat=zeros(nRCAmpBin,nPhaseBins);

% initialising binning plot for the loop
if checkBinningPlot
    nCol=ceil(nRCAmpBin/2);
    pIdx=1;
    figure
end

% phase binning for each amplitude
for ampRCBin=ampRCBinVect
    idxAmp=find(ampAtStimVect<ampRCBins(ampRCBin+1) & ampAtStimVect>ampRCBins(ampRCBin));
    
    if binningOption==0
        [phaseBins,phaseBinsCenter]=getBins(hilbertPhaseVectDeg(idxAmp),nPhaseBins);
    else
        %dubious meaning of a bin spanning -15 15 but got to use phasecodes for fit
        phaseBins1=[wrapTo180(phaseBinsCenter(end));phaseBinsCenter(1:end)];
        phaseBins2=[phaseBinsCenter(1:end);360];
        if useSimpleBins
            phaseBins = phaseBins2;
            phaseBinsCenter = phaseBinsCenter + 15;
        else
            phaseBins = (phaseBins1+phaseBins2)/2;
        end
        if phaseBins(1)>=0
            warning('pb in phase binning - the edge of the first phase bin should be negative')
        end
    end
    pRCBinsCenterMat(ampRCBin,:)=phaseBinsCenter(:)';
    for pBin=pRCBinVect
        if pBin==1 && ~useSimpleBins
            idxPh=find(hilbertPhaseVectDeg(idxAmp)<phaseBins(pBin+1) | (wrapTo360(phaseBins(pBin))>180 & hilbertPhaseVectDeg(idxAmp)>=wrapTo360(phaseBins(pBin))));
            %         if pBin==pRCBinVect(end)
            %             idxPh=find(hilbertPhaseVectDeg(idxAmp)<=phaseBins(pBin+1) & hilbertPhaseVectDeg(idxAmp)>=phaseBins(pBin));
        else
            idxPh=find(hilbertPhaseVectDeg(idxAmp)<phaseBins(pBin+1) & hilbertPhaseVectDeg(idxAmp)>=phaseBins(pBin));
        end
        deltaPhaseCell{ampRCBin,pBin}=deltaPhaseVect(idxAmp(idxPh));
        deltaAmpCell{ampRCBin,pBin}=deltaAmpVect(idxAmp(idxPh));
    end
    
    % binning plot
    if checkBinningPlot
        %                 figure
        %                 scatter(hilbertPhaseVectDeg(idxAmp),zeros(length(hilbertPhaseVectDeg(idxAmp)),1))
        %                 title(['amplitude idx ' num2str(ampRCBin)])
        subplot(2,nCol,pIdx)
        pIdx=pIdx+1;
        histogram(hilbertPhaseVectDeg(idxAmp),phaseBins)
        xlabel('phase (deg)')
        title(['amplitude idx ' num2str(ampRCBin)])
        xlim([0 360])
    end
    
end

% saving the binning plot (has to be outside of the loop)
if checkBinningPlot
    dimX=30;
    dimY=15;
    resolution='-r300';% does not change the size of plots, labels...
    changeFontBool=false;
    fontSize=[];
    changeLineThBool=false;
    lineTh=2;
    myDir='fig';
    mkdir(myDir);
    fileNameNoNowStr=[myDir '/binningCheck_' num2str(dataTag)];
    mySaveas( gcf, gca, dimX, dimY, resolution,...
        changeLineThBool, lineTh,...
        changeFontBool, fontSize,...
        fileNameNoNowStr );
    close
end

% applying the binning to get averages
if nPhaseBins==1
    avgFunc=@(x) mean(abs(x));
    stdFunc=@(x) std(abs(x));
else
    stdFunc=@std;
end

meanDeltaPhase=cellfun(avgFunc,deltaPhaseCell);%%%% changed to median
meanDeltaAmp=cellfun(avgFunc,deltaAmpCell);%%%% changed to median

stdDeltaPhase=cellfun(stdFunc,deltaPhaseCell);
stdDeltaAmp=cellfun(stdFunc,deltaAmpCell);
nSamples=cellfun(@length,deltaPhaseCell);
semDeltaPhase=stdDeltaPhase./sqrt(nSamples);
semDeltaAmp=stdDeltaAmp./sqrt(nSamples);%nSamples same for deltaPhaseCell and deltaAmpCell

minN=2;
toNaN=nSamples<minN;
meanDeltaPhase(toNaN)=NaN;
meanDeltaAmp(toNaN)=NaN;
semDeltaPhase(toNaN)=NaN;
semDeltaAmp(toNaN)=NaN;

% plot prc arc across for each amplitude in the same plot
if plotResponseCurves
    if plotForGihan
        
        set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
        set(groot, 'defaultLegendInterpreter','latex');
        
        figure
%         subplot(1,2,1);
        set(gcf,'color','w');
        hold on
        legendCell={};
        if plotOnlyExtremeBins
            binToPlot=[1,nRCAmpBin];
        elseif extremeAndMed
            binToPlot=[1,round(median(1:nRCAmpBin)),nRCAmpBin];
        else
            binToPlot=1:nRCAmpBin;
        end
        for i=binToPlot
            errorbar(deg2rad(pRCBinsCenterMat(i,:)),meanDeltaPhase(i,:),...
                semDeltaPhase(i,:),'linewidth',1.2)
%             if i==1
%                 legendCell=[legendCell,['Lowest amplitude bin (' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e') ')']];
%             elseif i==binToPlot(end)
%                 legendCell=[legendCell,['Highest amplitude bin (' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e') ')']];
%             else
%                 legendCell=[legendCell,['amp bin ' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e')]];
%             end
            if i==1
                legendCell=[legendCell,'lowest amplitude bin'];
            elseif i==binToPlot(end)
                legendCell=[legendCell,'high amplitude bin'];
            else
                legendCell=[legendCell,'medium amplitude bin'];
            end
        end
        legend(legendCell,'Location','best','FontSize',6);
        setLimMat(meanDeltaPhase,0.25)
        xlabel('$\psi$ (rad)','interpreter','latex')
        ylabel('$\Psi_e(\rho,\psi)$','interpreter','latex')
        savefig(['fig/PRC-' num2str(dataTag) '.fig'])
        
        figure
%         subplot(1,2,2);
        set(gcf,'color','w');
        hold on
        legendCell2={};
        for i=binToPlot
            errorbar(deg2rad(pRCBinsCenterMat(i,:)),meanDeltaAmp(i,:),...
                semDeltaAmp(i,:),'linewidth',1.2)
%             if i==1
%                 legendCell2=[legendCell2,['Lowest amplitude bin (' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e') ')']];
%             elseif i==binToPlot(end)
%                 legendCell2=[legendCell2,['Highest amplitude bin (' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e') ')']];
%             else
%                 legendCell2=[legendCell2,['amp bin ' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e')]];
%             end
            if i==1
                legendCell2=[legendCell2,'lowest amplitude bin'];
            elseif i==binToPlot(end)
                legendCell2=[legendCell2,'high amplitude bin'];
            else
                legendCell2=[legendCell2,'medium amplitude bin'];
            end
        end
        setLimMat(meanDeltaAmp,0.25)
        xlabel('$\psi$ (rad)','interpreter','latex')
        ylabel('$P_e(\rho, \psi)$','interpreter','latex')
        legend(legendCell2,'Location','best','FontSize',6);
        savefig(['fig/ARC-' num2str(dataTag) '.fig'])
              
%         dimX=30;
%         dimY=15;
%         resolution='-r300';% does not change the size of plots, labels...
%         changeFontBool=false;
%         fontSize=[];
%         changeLineThBool=false;
%         lineTh=2;
%         myDir='fig';
%         mkdir(myDir);
%         fileNameNoNowStr=[myDir '/RCs-' num2str(dataTag)];
%         mySaveas( gcf, gca, dimX, dimY, resolution,...
%             changeLineThBool, lineTh,...
%             changeFontBool, fontSize,...
%             fileNameNoNowStr );
%         close
  
        if saveForGihan
            p34.phaseBinsCenterMatRad=deg2rad(pRCBinsCenterMat);
            p34.ampBinsEdges=ampRCBins;
            p34.meanDeltaPhaseRad=meanDeltaPhase;
            p34.semDeltaPhaseRad=semDeltaPhase;
            p34.meanDeltaAmp=meanDeltaAmp;
            p34.semDeltaAmp=semDeltaAmp;
            
            fnameGihan=['patient34AmpBinning_' num2str(nRCAmpBin) 'bins'];
            
            save(fnameGihan,'p34');
        end
                  
    else
        figure
        subplot(1,2,1);
        set(gcf,'color','w');
        hold on
        legendCell={};
        if plotOnlyExtremeBins
            binToPlot=[1,nRCAmpBin];
        else
            binToPlot=1:nRCAmpBin;
        end
        for i=binToPlot
            errorbar(pRCBinsCenterMat(i,:),meanDeltaPhase(i,:),...
                semDeltaPhase(i,:),'o','linewidth',1.2)
            if i==1
                legendCell=[legendCell,['Lowest amplitude bin (' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e') ')']];
            elseif i==binToPlot(end)
                legendCell=[legendCell,['Highest amplitude bin (' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e') ')']];
            else
                legendCell=[legendCell,['Amplitude bin ' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e')]];
            end
        end
        legend(legendCell,'Location','best','FontSize',6);
        setLimMat(meanDeltaPhase,0.25)
        xlabel('Stimulation phase (deg)')
        ylabel('Change in phase (rad)')
        title('PRC')
        
        subplot(1,2,2);
        set(gcf,'color','w');
        hold on
        legendCell2={};
        for i=binToPlot
            errorbar(pRCBinsCenterMat(i,:),meanDeltaAmp(i,:),...
                semDeltaAmp(i,:),'o','linewidth',1.2)
            if i==1
                legendCell2=[legendCell2,['Lowest amplitude bin (' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e') ')']];
            elseif i==binToPlot(end)
                legendCell2=[legendCell2,['Highest amplitude bin (' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e') ')']];
            else
                legendCell2=[legendCell2,['Amplitude bin ' num2str(ampRCBins(i),'%.2e') ' to ' num2str(ampRCBins(i+1),'%.2e')]];
            end
        end
        setLimMat(meanDeltaAmp,0.25)
        xlabel('Stimulation phase (deg)')
        ylabel('Change in amplitude (au)')
        title('ARC')
        legend(legendCell2,'Location','best','FontSize',6);
        globalTitle=['Patient ' strrep(num2str(dataTag),'_','\_') ' - ' num2str(length(hilbertPhaseVect)) ' data points'];
        suptitle({globalTitle,[num2str(nRCAmpBin) ' amplitude bins, ' num2str(nPhaseBins) ' phase bins']})%,['noise = ' num2str(dat.noise) ' - coupling = ' num2str(dat.coupling)]})
        
        dimX=30;
        dimY=15;
        resolution='-r300';% does not change the size of plots, labels...
        changeFontBool=false;
        fontSize=[];
        changeLineThBool=false;
        lineTh=2;
        myDir='fig';
        mkdir(myDir);
        fileNameNoNowStr=[myDir '/RCs-' num2str(dataTag)];
        mySaveas( gcf, gca, dimX, dimY, resolution,...
            changeLineThBool, lineTh,...
            changeFontBool, fontSize,...
            fileNameNoNowStr );
        close
    end
    
end

% RC object to return
RCs.pRCBinsCenterMat=pRCBinsCenterMat;
RCs.meanDeltaPhase=meanDeltaPhase;
RCs.semDeltaPhase=semDeltaPhase;
RCs.meanDeltaAmp=meanDeltaAmp;
RCs.semDeltaAmp=semDeltaAmp;
RCs.ampRCBins=ampRCBins;
RCs.numOfPulsesForNorm=numOfPulsesForNorm;

% response surfaces
X=pRCBinsCenterMat; %if x axis is phases and y amp, X mat should be rows of phases (nAmp x nPh)
ampBinCenters=(ampRCBins(1:end-1)+ampRCBins(2:end))/2;
Y=repmat(ampBinCenters',1,nPhaseBins); %Y mat should be columns of amp (nAmp x nPh)
Zph=meanDeltaPhase; %Z also nAmp x nPh
Zamp=meanDeltaAmp; %Z also nAmp x nPh

if plotSurface
    my3dSubPlot4Views(X,Y,Zph,...
        'stimulation phase (deg)',...
        'amplitude (a.u.)',...
        'change in phase (rad)',...
        'phase response surface from amplitude binning',...
        false); %save surf
    
    my3dSubPlot4Views(X,Y,Zamp,...
        'stimulation phase (deg)',...
        'amplitude (a.u.)',...
        'change in amplitude (a.u.)',...
        'amplitude response surface from amplitude binning',...
        false); %save surf
end

surfFromBin.X=X;
surfFromBin.Y=Y;
surfFromBin.Zph=Zph;
surfFromBin.ZphSem=semDeltaPhase;
surfFromBin.ZphStd=stdDeltaPhase;
surfFromBin.Zamp=Zamp;
surfFromBin.ZampSem=semDeltaAmp;
surfFromBin.ZampStd=stdDeltaAmp;
surfFromBin.info=dataTag;
surfFromBin.binningOption=binningOption;
%         surfFromBin.synthDataId=fname;
surfFromBin.numOfPulsesForNorm=numOfPulsesForNorm;

if nPhaseBins==1
    figure
    errorbar(surfFromBin.Y,surfFromBin.Zph,surfFromBin.ZphSem)
    ylabel('change in phase')

    figure
    errorbar(surfFromBin.Y,surfFromBin.Zamp,surfFromBin.ZampSem)
    ylabel('change in amp')
end

end

%%% code for boxplot if needed at some point.
%     group=[];
%     deltaPhForBx=[];
%     deltaAmpForBx=[];
%
%     for i=1:length(nSamples)
%         group=[group;repmat({[num2str(round(phaseBinsCenter(i))) '°']},...
%             nSamples(i),1)];
%         deltaPhForBx=[deltaPhForBx;deltaPhaseBin{i}];
%         deltaAmpForBx=[deltaAmpForBx;deltaAmpBin{i}];
%     end
%
%     figure
%     boxplot(deltaPhForBx,group)
%     title('PRC - single pulse')