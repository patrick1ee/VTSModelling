function [ yPSDmean ] = getTrAmpPSD( sig, env, fs, trialIdx, freqRange, wLratio, verbose )
%getting the amplitude pdf of sig averaged over the trials given in trialIdx

assert(~isempty(sig) || ~isempty(env),'need to supply signal or envelope');% throw error if condition false

if isempty(env)
    env = abs(hilbert(sig));
end

% env=detrend(env);%detrending by block needed?

nTr=size(trialIdx,1);

yPSD=zeros(nTr,length(freqRange));

for tr=1:nTr%8
    psdIdx=trialIdx(tr,1):trialIdx(tr,2);
    wLength=round(length(env(psdIdx))/wLratio);
    yPSDtr=pwelch(env(psdIdx),wLength,[],freqRange,fs);
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
    ylabel('amplitude PSD')
    xlabel('frequency (Hz)')
    legend(gca,'show')
end

end
