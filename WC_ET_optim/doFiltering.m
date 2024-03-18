function [ filteredSignal ] = doFiltering( params,signal )
% no phase distortion bandpass filtering 
% signal -> time series to filter
% params -> structure with the following fields:
%    dt: sampling interval
%    fcutlow and fcuthigh: frequency bounds of the filtering band

fs=1/params.dt;
[b,a]    = butter(params.filterOrder,[params.fcutlow,params.fcuthigh]/(fs/2), 'bandpass');
filteredSignal        = filtfilt(b,a,signal);

end

