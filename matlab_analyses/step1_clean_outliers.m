function [] = step1_clean_outliers(Paths, Const)

% USER INSTRUCTION: 
% To skip one condition forwards click above the panel
% To skip one condition backwards click below the panel
% To stop the cleaning procedure, click into the bottom "Command Window" of
% the Matlab editor and press CTRL+C, and then close the window

VERTICAL_CHAN_SHIFT = 0.0001;
condLabel           = Const.condLabel;
START_FROM_SCRATCH  = false;


for fn = Const.participants
    participant_ID = ['P', num2str(fn)];

   load([Paths.matfiles, participant_ID, '.mat'], 'wvData') % save the data as matfile to compute group statistics
   fnames = fieldnames(wvData);
   conditions = fnames;
   idxRem = contains(conditions, 'fltBand')
   conditions(idxRem) = [];

    outliers_loaded = false;       
    if ~START_FROM_SCRATCH
       try 
            load([Paths.matfiles, participant_ID, '_outliers'], 'outliers') % save the data as matfile to compute group statistics
            outlier_fnames = fieldnames(outliers);
            outliers_loaded = true;
       end
    end
    c = 1;
    while c <= numel(conditions)
        clear eeg_data
        currCond = conditions{c};
        c = c + 1;
        try
            if isnan(wvData.(currCond).EEGc) % skip if not data is present
                continue
            end
        end
        if ~outliers_loaded || ~sum(strcmp(outlier_fnames, currCond))
            outliers.(currCond) = false(size(wvData.(currCond).EEGc));
        end
        
        fig_h = figure(1);
        set(fig_h, 'Windowstate', 'maximized')
        titling = [participant_ID, ' ', currCond];
        title(strrep(titling, '_', ' '))
        eeg_data(1,:) = wvData.(currCond).EEGc;
        eeg_data(2,:) = wvData.(currCond).EEGi;
        if outliers_loaded
            try 
                eeg_data(:, outliers.(currCond)) = nan;
            end
        end
        xIdcs = 1:size(eeg_data,2);
        for chan = 1:size(eeg_data,1)  % Plot all the EEG channels
            plot(eeg_data(chan, xIdcs) - VERTICAL_CHAN_SHIFT * chan, 'Color', 0.2*[1,1,1]); hold on
        end
        yLims = get(gca,'YLim');
        axis([1 numel(xIdcs), yLims])        

        doLoop = true;
        firstCoord = true;
        while doLoop
            drawnow
            try
                [x,y] = ginput(1);  % gathers points until I press the return key
            catch
                break
            end
            if x < 0  % if you clicked beyond the left border of the image
               x = 1; % set x to 1 (the first index)
            end
            if y > yLims(2) % if you click above the panel, then stop the cleaning procedure
                break
            end
            if y < yLims(1) 
                c = c - 2; % Go back one step
                break
            end
            
            if firstCoord  % This is needed because we always want to read in two mouseclicks, the 
                % first indicates the start of the window that includes the
                % artfact, and the second click indicates the end of this
                % window
                firstCoord = false;
                x1 = round(x);
                x2 = nan;  
            else
                x2 = round(x);                  
%                 if ~isnan(x2)                  
                if x2 > numel(xIdcs), x2 = numel(xIdcs); end % if you clicked outside the right border, set it to the last (rightmost) index
                
                currIdcs = (x1 : x2);  % to tag the correct part in the EEG signal you need to consider/add the offset 

                eeg_data(:, currIdcs)  = nan;
                outliers.(currCond)(currIdcs) = true;
                clf   % need to clear the figure before plotting it again without the outliers

                for chan = 1:size(eeg_data,1)  % Plot all the EEG channels
                    plot(eeg_data(chan,xIdcs) - VERTICAL_CHAN_SHIFT * chan, 'Color', 0.2*[1,1,1]); hold on
                end
                title(strrep(titling, '_', ' '))

                xlim([1,numel(xIdcs)])
                yLims = get(gca,'YLim');
%                 end
                firstCoord = true;  % need to set this to true again, to be able to tag another window with artefacts (using again two mouseclicks)
            end  
        end
        clf
        save([Paths.matfiles, participant_ID, '_outliers'], 'outliers') % save the data as matfile to compute group statistics
   end
end