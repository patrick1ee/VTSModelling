function [ yPDFmean ] = getTrAmpPDF( sig, env, trialIdx, xAmp, verbose )
%getting the amplitude pdf of sig averaged over the trials given in trialIdx

assert(~isempty(sig) || ~isempty(env),'need to supply signal or envelope');% throw error if condition false

if isempty(env)
    env = abs(hilbert(sig));
end

nTr=size(trialIdx,1);

yPDF=zeros(nTr,length(xAmp));

for tr=1:nTr%8
    trIdx=trialIdx(tr,1):trialIdx(tr,2);
    yPDFtr = ksdensity(env(trIdx),xAmp);
    yPDF(tr,:)=yPDFtr;
end

yPDFmean=mean(yPDF,1);

if verbose
    figure
    hold on
    for tr=1:nTr
        plot(xAmp,yPDF(tr,:),'DisplayName',['trial #' num2str(tr)])
    end
    plot(xAmp,yPDFmean,'linewidth',3.5,'DisplayName','average')
    ylabel('PDF')
    xlabel('amplitude')
    legend(gca,'show')
end

end

