function [E,I,stimVect]=...
    doEulerNoiseTrackNStimCluster(wIE,wEI,wEE,beta,Tau,thetaE,thetaI,nMax,sigma,dt,stim_prog,delayMs,stimInSigmoid)
%29-11-18  	seeding added
% v2: delay, 6 pulses and direct stim or signoid stim

nPulses = 6 ;
fPulses = 130 ;
idxPulses = round(1/fPulses/dt);

%converting delay in ms to idx
delay=round(delayMs*1E-3/dt);

E=zeros(nMax,1);%zeros(ceil(nMax),1);% strange matlab behaviour...
Edot=zeros(nMax,1);

I=zeros(nMax,1);
Idot=zeros(nMax,1);

stimVect=zeros(nMax,1);
% phaseVect=zeros(nMax,1);%for debug
% xingIdxVect=[];%for debug
% meanVect=zeros(nMax,1);%for debug

E(1)=0;
I(1)=0;

%set hysteresis level based on noise std dev - maybe initially, then periodically adjust it based on generated signal?
th=0.6*sigma;%0.6

% eps=deg2rad(5);%1 deg

starting=true;
missed=false;
stimDone=false;
f=5;%initial value used for starting things up
% nf=1;%how many previous freq to average over
%oscillations between nMean and nDoMean will be removed
nMean=round(20/f/dt);%conversion of number of oscillations for mean
nDoMean=round(60/f/dt);%conversion of number of oscillations before looking at the mean
rng('shuffle');  
randMat=normrnd(0,1,2,nMax-1);

Emean=0;
ref = 0;
pulseCount = 0;

% nReDoMean=5;
% nFirstReDo=round(nMax/nReDoMean);
% Euler method
for i=1:nMax-1
    
    %%% Re recalculate the mean a certain number of times.
    % does not seem useful...
%     if mod(i,nFirstReDo)==0
%         Emean=mean(E(i-nMean:i));
%         EE=E(i)-Emean;
%     end
    
    if i<nDoMean
        EE=E(i)-mean(E(max(1,i-nMean):i));% could probably be made faster -> becomes prohibitive for large n
    elseif Emean==0
        Emean=mean(E(i-nMean:i));
        EE=E(i)-Emean;
        th=0.2*std(E(i-nMean:i));%0.12
    else
        EE=E(i)-Emean;
    end
        
%     meanVect(i)=EE;%%% for debug
    stimMag=0;
    
    if starting
        lastXingIdx=1;%delay?
        if EE>th 
            pos=true;
            starting=false;
        elseif EE<-th 
            pos=false;
            starting=false;
        end
    %current phase
    elseif pos  
        if EE<-th
            pos=false;
            lastBeforeThIdx=i;
        end
        phase=(i-lastXingIdx)*dt*f*2*pi;%in deg, evolution according to f of previous period
        if phase>=2*pi
            phase=0;
        end
%         phaseVect(i)=phase;%%% for debug
    else  
        if EE>th
            pos=true;       
            if exist('lastBeforeThIdx','var')
                newXingIdx=(i+lastBeforeThIdx)/2;% removing round
            else
                newXingIdx=i;
            end
%             xingIdxVect=[xingIdxVect,round(newXingIdx)]; %%% for debug
%             f=[1/((newXingIdx-lastXingIdx)*dt),f];
            f=1/((newXingIdx-lastXingIdx)*dt);
%             if length(f)>nf
%                 f(end)=[];
%             end
            lastXingIdx=newXingIdx;
            phase=(i-newXingIdx)*dt*f*2*pi;
%             phaseVect(i)=phase;%%% for debug
            if ~stimDone
                missed=true;
            end
            stimDone=false; %resetting the trigger at the very beginning of a period 
        else
            phase=(i-lastXingIdx)*dt*f*2*pi;%in deg, evolution according to f of previous period
            if phase>=2*pi
                phase=0;
            end     
%             phaseVect(i)=phase;%%% for debug      
            if EE<=-th
                lastBeforeThIdx=i;
            end
        end
    end
    
delayIdx=max(1,i-delay);
    
    if stim_prog(delayIdx,1)==0
        missed=false; % otherwise missed will cause an unexpected firing as soon as a block starts
        stimDone=true; % same thing, because of the way the condition on phase for stim is written
    end
    
    if stim_prog(delayIdx,1)~=0 &&...
        ~stimDone && ...
        ((missed || stim_prog(delayIdx,2)<phase))
    
        stimMag=stim_prog(delayIdx,1);
        stimVect(delayIdx)=stimMag;
        ref=i;
        pulseCount=nPulses-1;
        
        if ~missed % stimDone should stay false if only firing a missed pulse at the very beginning of a period
            stimDone=true;
        end
        missed=false;
        
    end
    
   
    if i~=ref && pulseCount > 0 && mod(i-ref,idxPulses)==0
        stimMag=stim_prog(delayIdx,1);
        stimVect(delayIdx)=stimMag;
        pulseCount=pulseCount-1;
    end
      
    randS = randMat(1,i);
    randG = randMat(2,i);
    
    if stimInSigmoid
        
    Edot(i)=(-E(i)+1/(1+exp(-beta*(thetaE-wIE*I(i)+wEE*E(i)+stimMag-1))))/Tau+sigma*randS/sqrt(dt);
    Idot(i)=(-I(i)+1/(1+exp(-beta*(thetaI+wEI*E(i)-1))))/Tau+sigma*randG/sqrt(dt);
    E(i+1)=E(i)+Edot(i)*dt;
    I(i+1)=I(i)+Idot(i)*dt;
   
    else
        
    Edot(i)=(-E(i)+1/(1+exp(-beta*(thetaE-wIE*I(i)+wEE*E(i)-1))))/Tau+sigma*randS/sqrt(dt);
    Idot(i)=(-I(i)+1/(1+exp(-beta*(thetaI+wEI*E(i)-1))))/Tau+sigma*randG/sqrt(dt);
    E(i+1)=E(i)+stimMag+Edot(i)*dt;
    I(i+1)=I(i)+Idot(i)*dt;

    end
    


end

end


