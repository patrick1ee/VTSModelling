function  [trialsMatrix, trialsVect, idcsMatrix] = getTrialsMatrix(events, data, dtTime, tWidth, tOffset)
% GETTRIALSMATRIX splits continuous data into epochs around events with a width 
% tWidth and an offset tOffset (positive numbers shift the epoch to start
% earlier than the event, negative numbers shift it to start after the
% event by the set offset), works both for time-frequency and time-data only
% OUTPUTS 
% trialsMatrix - matrix with epochs (time*freq*trials) if data is a
%               (time*freq) matrix, or (time*trials) if data is a time-vector      

events     = events(~isnan(events));

nEvent     = length(events);
erspHeight = size(data,1);

nWidth  = round(tWidth/dtTime);      
nOffset = round(tOffset/dtTime); 

trialsMatrix = nan(erspHeight, nWidth + 1, nEvent);
idcsMatrix   = nan(erspHeight, nWidth + 1, nEvent);
trialsVect   = nan(size(data));
idcsVect     = 1:size(data,2);

for i = 1:nEvent
    if isnan(events(i)), continue; end
    currtime = round(events(i)/dtTime);
    n1 = currtime - nOffset;
    n2 = n1 + nWidth;         
    trialsMatrix(:, :, i) = data(:, n1:n2);
    idcsMatrix(:,:,i) = repmat(idcsVect(:, n1:n2), size(idcsMatrix,1), 1, 1 );
    if size(data,1) == 1  % don't compute this if you receive a freq*time matrix
%         x = n2-n1
        trialsVect(n1:n2) = data(:, n1:n2);
    end
end

trialsMatrix = squeeze(trialsMatrix);
idcsMatrix = squeeze(idcsMatrix);