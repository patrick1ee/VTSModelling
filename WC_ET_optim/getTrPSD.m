function [ yPSDmean ] = getTrPSD( sig, fs, trialIdx, freqRange, wLratio, verbose )
%getting the psd of sig averaged over the trials given in trialIdx

nTr=size(trialIdx,1);
yPSD=zeros(nTr,length(freqRange));

for tr=1:nTr%8
    psdIdx=trialIdx(tr,1):trialIdx(tr,2);
    wLength=round(length(sig(psdIdx))/wLratio);
    yPSDtr=pwelch(sig(psdIdx),wLength,[],freqRange,fs);
    yPSD(tr,:)=yPSDtr;
end

yPSDmean=mean(yPSD,1);

if verbose
    figure
    hold on
    for tr=1:nTr
        plot(freqRange,yPSD(tr,:),'DisplayName',['trial #' num2str(tr)])
    end
    plot(freqRange,yPSDmean,'linewidth',3.5,'DisplayName','average')
    ylabel('PSD')
    xlabel('frequency (Hz)')
    legend(gca,'show')
end

end

