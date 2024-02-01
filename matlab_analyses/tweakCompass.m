function[] = tweakCompass(gcf, currFontSize)

th = findall(gcf,'Type','text');
for i = 1:length(th)
    set(th(i),'FontSize',currFontSize-2)
    if ~(i==1      || i==2      || i==7      || i==8 )
        set(th(i),'String','')
    end
%     if ~(i==1      || i==2      || i==7      || i==8 || ...
%          i==1+17   || i==2+17   || i==7+17   || i==8+17 || ...           
%          i==15     || i==15+17   )
%         set(th(i),'String','')
%     end
end

% Change labels from degrees to radians
labelPos = [2,          7,              8];
% radLabel = {'\pi',     '3\pi/2',   '\pi/2'};
radLabel = {'180',     '270/-90',   '90'};
% labelPos = [2, 2+17,          7, 7+17,              8, 8+17];
% radLabel = {'\pi', '\pi',    '3\pi/2', '3\pi/2',   '\pi/2', '\pi/2'};
for lp = 1:numel(labelPos)
    set(th(i),'FontSize',currFontSize-2)
    set(th(labelPos(lp)), 'String', radLabel{lp})
end
