function optimPatternStandAlone(folderName, J, K, taskFactor, dataId, includeShift, nTr, PSDtolX, PSDtolY, restrictToFP, maxFcnCalls, meshTol, df, nNaNtest, useBounds, doScaling)
%%% 23-12-18    including lb and ub for patternsearch
%%%             possibility of scaling
%%% 18-12-18    NaNtest to avoid false starts of patternsearch 
%%%             K needed for filename
%%% 17-12-18    with pattern search
%%% 02-12-18    param gen for each worker individually 
%%%             param rejection if PSD is not within PSDtolx PSDtolY of data (eg. below)
%%% 29-11-18	tighter prior again
%%% 18-11-18	same prior as snpe (restrictToOscil already false)
%%% 18-05-18	conversion to start an optimisation from scratch

% tolX = 2; %Hz
% tolY = 0.5; %pct, 1 = 100 %

tic
mkdir(folderName);
diary([folderName '/diary_' folderName])

if nargin~=16
    error(' *** sixteen arguments are expected')
end

if isdeployed
    J=str2num(J);
    K = str2num(K);
    taskFactor=str2num(taskFactor);
    dataId=str2num(dataId);
    restrictToFP = strcmp(restrictToFP,'true');
    includeShift = strcmp(includeShift,'true');
    nTr = str2num(nTr);
    PSDtolX=str2num(PSDtolX);
    PSDtolY=str2num(PSDtolY);
    maxFcnCalls=str2num(maxFcnCalls);
    meshTol=str2num(meshTol);
    df=str2num(df);
    nNaNtest = str2num(nNaNtest); 
    useBounds = strcmp(useBounds,'true');
    doScaling = strcmp(doScaling,'true');
end

temp=load(['dataFeatures-patient' num2str(dataId)]);
dataFeatures=temp.dataFeatures;

%% Optimisation
nTasks = 1;

%%%%%%%%%%%%%%%%%
nLoc = nTasks*taskFactor;
stimInSigmoid = false;
% maxFcnCalls = 1000;
% meshTol = 1E-4;
%%%%%%%%%%%%%%%%%

rng('shuffle');

options = optimoptions('patternsearch','MeshTolerance',meshTol,'MaxFunctionEvaluations',maxFcnCalls);
% options = optimoptions('patternsearch','MeshTolerance',meshTol,'MaxFunctionEvaluations',maxFcnCalls,'Display','iter','PlotFcn',{'psplotbestf','psplotmeshsize','psplotfuncount'});

if useBounds
    wIElim = [0 30];
    wEIlim = [0 30];
    wEElim = [0 30];
    betaLim = [0 30];
    xqLim = [-30 30];    
    yqLim = [-30 30];
    noiseLim = [0 0.3];
    tauLim = [0 0.5];
    stimMagLim = [0 0.1];
    delayMsLim = [0 500];

    lb = [wIElim(1),wEIlim(1),wEElim(1),betaLim(1),tauLim(1),xqLim(1),yqLim(1),noiseLim(1),stimMagLim(1),delayMsLim(1)];
    ub = [wIElim(2),wEIlim(2),wEElim(2),betaLim(2),tauLim(2),xqLim(2),yqLim(2),noiseLim(2),stimMagLim(2),delayMsLim(2)];
else
    lb = [];
    ub = [];
end

if doScaling 
    
    wIEscale = 5;
    wEIscale = 5;
    wEEscale = 5;
    betaScale = 5;%could be reduced
    xqScale = 5;      
    yqScale = 5;
    noiseScale = 0.05;
    tauScale = 0.15;
    stimMagScale = 0.01;
    delayMsScale = 125;
    scale = [wIEscale,wEIscale,wEEscale,betaScale,tauScale,xqScale,yqScale,noiseScale,stimMagScale,delayMsScale];
    
    lb = lb ./ scale;
    ub = ub ./ scale;
end

for i=1:nLoc
    NaNtest = ones(nNaNtest,1);%needed because of the parfor
    PSDok = false;
    while ~PSDok
        wIE = uniRand(0,10);
        wEI = uniRand(0,10);
        wEE = uniRand(0,10);
        beta = uniRand(0,10);%could be reduced
        xq = uniRand(-2,10);      
        yq = uniRand(-10,2);
        noise = uniRand(0,0.1);
        tau = uniRand(0,0.3);
        stimMag = uniRand(0,0.02);
        delayMs = uniRand(0,250);
        
%         wIE = rand(1,1)*10;uni
%         wEI = rand(1,1)*10;
%         wEE = rand(1,1)*10;
%         beta = rand(1,1)*10;
%         xq = rand(1,1)*20-10;
%         yq = rand(1,1)*20-10;
%         noise=rand(1,1)*0.15;
%         stimMag=rand(1,1)*0.02;%0.1;
%         delayMs=rand(1,1)*250;
%         tau = rand(1,1)*1E-2+4E-2;

        tentativeParams = [wIE,wEI,wEE,beta,tau,xq,yq,noise,stimMag,delayMs];
        PSDok = isPSDok(tentativeParams, dataFeatures, PSDtolX, PSDtolY, false);
        if restrictToFP && PSDok
            PSDok = isFP(tentativeParams, false);% verbose only if out (see function)
        end
        if nNaNtest>0 && PSDok 
            for nn = 1:nNaNtest
                NaNtest(nn) = isnan(costPattern(tentativeParams, dataFeatures, '', stimInSigmoid, includeShift, nTr, df));
            end
            PSDok = (sum(NaNtest) == 0);
        end
    end

    filename=[folderName '/' num2str(J) '_' num2str(K) '_' num2str(i) '.txt'];
    fileID = fopen(filename,'a');
    fprintf(fileID,'%12s %12s %12s %12s %15s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\r\n',...
		'wIE','wEI','wEE','beta','Tau','thetaE','thetaI','sigma','stimMag','delay','cost','cost1','cost2','cost3','cost4','cost5','fCenter','maxPSD','Estd','nNaN');
    fclose(fileID);
    
    if doScaling
        singleArgCostFunc = @(x) costPattern(x.*scale, dataFeatures, filename, stimInSigmoid, includeShift, nTr, df);
    else
        singleArgCostFunc = @(x) costPattern(x, dataFeatures, filename, stimInSigmoid, includeShift, nTr, df);
    end
    
    try
        if doScaling
            [x,fval,exitflag,output] = patternsearch(singleArgCostFunc,tentativeParams./scale,[],[],[],[],lb,ub,[],options)  
        else
            [x,fval,exitflag,output] = patternsearch(singleArgCostFunc,tentativeParams,[],[],[],[],lb,ub,[],options)  
        end
        
    catch err
	    warning(['local optim ' num2str(i) ' crashed']);
	    disp(getReport(err,'extended'));
    end

end

toc
diary off
exit
end

