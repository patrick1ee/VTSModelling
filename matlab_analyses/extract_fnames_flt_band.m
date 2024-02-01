function [fileNames, shortNames, flt_band] = extract_fnames_flt_band(Paths, Const, currSubj, vers)


eyesStatus = 'EO'; % 'EO', for P1
% eyesStatus = 'EC'; % 'EO', for rest data 

%% Automatically extract filenames from folder - this can have issues under MAC:
folder_info = dir(fullfile([Paths.data, currSubj]));
all_fnames  = {folder_info.name};

% First find noStim condition
findCond = strfind(all_fnames(:), 'NOSTIM');
findVersion = strfind(all_fnames(:), vers);
findEyesClosed = strfind(all_fnames(:), eyesStatus);
getIdx = ~cellfun(@isempty, findCond) & ~cellfun(@isempty, findVersion) & ~cellfun(@isempty, findEyesClosed);
if sum(getIdx)==0
    error(['Make sure a noStim ', vers ' file is present in ', Paths.data, currSubj])
end
fileNames(1) = all_fnames(getIdx);
shortNames{1} = [' noStim'];

% Then use a for-loop to run through all phases and check if files are
% available
for ph = 1:numel(Const.phases)
    currPhase = ['phase=', num2str(Const.phases(ph)), '_STIM'];
    findCond = strfind(all_fnames(:), currPhase);
    findVersion = strfind(all_fnames(:), vers);

    getIdx = ~cellfun(@isempty, findCond) & ~cellfun(@isempty, findVersion);
    if sum(getIdx)>0
        fileNames(end+1) = all_fnames(getIdx);      
        shortNames{end+1} = [num2str(Const.phases(ph))];
    end
end
% Automatically extract the target frequency from the filename
lastFile = fileNames{end};
startNum = strfind(lastFile, 'Q=');
endNum = strfind(lastFile, 'Hz');
flt_band = str2num(lastFile(startNum+2:endNum-1)) + [-1, +1]; % Hz
