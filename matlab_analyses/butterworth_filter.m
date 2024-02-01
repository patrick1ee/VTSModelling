function  [fltData] =  butterworth_filter(data,  FLT_BAND, FLT_ORDER, FLT_DIRECTION, SR)

orig_Length = numel(data);
% First mirror the data to avoid edge artefacts
% which avoids sharp edges on each side and improves the output of the filtering
data   = [flipud(data); data; flipud(data)];

% Butterworth filter
[b,a] = butter(FLT_ORDER, FLT_BAND(1)/(SR/2), 'high');
data_flt1 = filter(b, a, data);
if strcmp(FLT_DIRECTION, 'twopass') % the default is a onepass filter
    data_flt1 = filtfilt(b, a, data);
end
[b,a]  = butter(FLT_ORDER, FLT_BAND(2)/(SR/2), 'low');
fltData = filter(b, a, data_flt1);
if strcmp(FLT_DIRECTION, 'twopass')
    fltData = filtfilt(b, a, data_flt1);
end

% Remove the appended edges again and return it
fltData = fltData(orig_Length+1:orig_Length*2);