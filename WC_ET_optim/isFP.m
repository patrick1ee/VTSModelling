function out = isFP (params, verbose)
% creation      12-12-18
% outputs true if the model corresponds to a stable fixed point

tolE = 5E-3;
tolI = 5E-3;

wIE=params(1);
wEI=params(2);
wEE=params(3);
beta=params(4);
Tau=params(5);
thetaE=params(6);
thetaI=params(7);

% locating the fixed point, finding the eigenvalues of the Jacobian
%%% Useful for confirming the type of fixed point + omega0

f=@(x)(1/(1+exp(-beta*(x-1))));
fprime=@(x)(beta*f(x)*(1-f(x)));
root2d=@(x)[-x(1)+f(thetaE+wEE*x(1)-wIE*x(2)),-x(2)+f(thetaI+wEI*x(1))];

%fixed point
%let the system evolve for a long time in the case of a stable fixed point to get an estimate.
dt = 1E-3;
N = round(200/dt);
[E,I]=doEulerX0Mex(wIE,wEI,wEE,beta,Tau,thetaE,thetaI,N,dt,0.5,0.5);

%last 3 periods at 5 Hz
idx = (N - round(0.2*3/dt)):N;

if max(E(idx))-min(E(idx)) < tolE && max(I(idx))-min(I(idx)) < tolI
    
    x0 = [E(end),I(end)];
    options = optimset('Display','off'); 
    S = fsolve(root2d,x0,options);
    Estar=S(1);
    Istar=S(2);

    %eigenvalues of Jacobian
    L=1/Tau*[-1+wEE*fprime(thetaE+wEE*Estar-wIE*Istar),-wIE*fprime(thetaE+wEE*Estar-wIE*Istar);...
        wEI*fprime(thetaI+wEI*Estar),-1];
    [~,D,~] = eig(L);

    sigma=real(D(1,1));
    out = real(sigma) < 0;
else
    out = false;
end

if verbose && out
    figure
    hold on
    plot(E,I)
    plot(E,I)
    ylabel('I')
    xlabel('E')
    title(['is FP = ' num2str(out)])
end

end