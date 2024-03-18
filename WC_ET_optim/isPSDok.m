function out = isPSDok (params, dataFeatures, tolX, tolY, verbose)
% creation      02-12-18
% outputs true if the model PSD peak is within tol of the data PSD

% tolX = 2; %Hz
% tolY = 0.5; %pct, 1 = 100 %

wIE=params(1);
wEI=params(2);
wEE=params(3);
beta=params(4);
Tau=params(5);
thetaE=params(6);
thetaI=params(7);
sigma=abs(params(8));
dt=1E-3;

tmax = 100 / 5; %about 100 periods
N = round(tmax / dt);
E = doEulerNoiseX0Mex(wIE,wEI,wEE,beta,Tau,thetaE,thetaI,N,sigma,dt,0,0);

rampUpN = 0.2 * N;

Emean=mean(E(rampUpN:end));
Estd=std(E(rampUpN:end));
Eproc=(E-Emean)/Estd;
% Eproc=detrend(Eproc);

EforPSD = Eproc(rampUpN:end);
freqRange = dataFeatures.PSD.xPSD;
yPSDmod = pwelch(EforPSD,round(length(EforPSD)/dataFeatures.PSD.wLratio),[],freqRange,1/dt);

[maxMod,iMod] = max (yPSDmod);
[maxDat,iDat] = max (dataFeatures.PSD.yPSD);
xMod = freqRange(iMod);
xDat = freqRange(iDat);


out = abs(xDat - xMod) < tolX && abs(maxDat - maxMod) < tolY * maxDat;

if verbose && out
    figure
    hold on
    plot(freqRange,yPSDmod,'linewidth',2,'displayName','model')
    plot(freqRange,dataFeatures.PSD.yPSD,'displayName','data')
    scatter(xMod,maxMod,'displayName','model')
    scatter(xDat,maxDat,'displayName','data')
    ylabel('PSD')
    xlabel('frequency (Hz)')
end

end