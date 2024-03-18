function sig_filtered = filteringProcess(sig,fs,bandSpecified,fminus,fplus,aroundMean,dfminus,dfplus,verbose)
%%% wrapper for doFiltering
% sig -> signal to be filtered
% fs -> sampling frequency
% bandSpecified -> bool, set to true to specify directly the frequency bounds
% of the bandpass filter (in fplus and fminus)
% aroundMean -> if bounds not directly specify, set to true to center the 
% bandpass filter around the mean frequency (energy), or to false to center 
% the bandpass on the peak frequency. The filtering window will be 
% [center-dfminus,center+dfplus].
% verbose -> set to true to display a plot of the raw and filtered signals

NFFT = length(sig);
F=(0:1:NFFT-1)*fs/NFFT; % better formulation (no rounding error)
fft_peak = abs(fft(sig)).^2;

params.dt = 1/fs;
params.filterOrder = 2;

if bandSpecified
    params.fcutlow  = fminus;
    params.fcuthigh = fplus;
else
    if ~aroundMean
        %%% mode between 3 and 100 Hz
        [~,start]=min(abs(F-3));
        [~,stop]=min(abs(F-100));
        [~,idx]=max(fft_peak(start:stop));
        peakFreq = F(idx+start-1);
    else
        %%% weighted average between 1 and 20 Hz
        [~,start]=min(abs(F-1)); 
        [~,stop]=min(abs(F-20)); 
        peakFreq=sum(F(start:stop).*fft_peak(start:stop))/sum(fft_peak(start:stop));
    end
    params.fcutlow  = peakFreq-dfminus;
    params.fcuthigh = peakFreq+dfplus; 
end

sig_filtered = doFiltering(params,sig);

%%% plotting
if verbose
    fft_raw= abs(fft(sig)).^2;
    fft_filtered= abs(fft(sig_filtered)).^2;
    
    figure
    h1=subplot(2,1,1);
    hold on
    plot(F,fft_raw)
    legend('raw')
    
    h2=subplot(2,1,2);
    plot(F,fft_filtered)
    legend('filtered')
    linkaxes([h1,h2], 'xy')
    
    xlim([0 30])
    ylim([0 max(fft_filtered)])
end

end

