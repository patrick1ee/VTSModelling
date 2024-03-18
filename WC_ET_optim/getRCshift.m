function shift = getRCshift(xPRC,yPRC,xARC,yARC)
% creation      28-11-18
% to include in cost func

    cosMod = @(b, x) b(1)+abs(b(2))*cos(x*pi/180+b(3));
    b0_2p=[nanmean(yPRC); (nanmax(yPRC)-nanmin(yPRC)); 0];
    b0_2a=[nanmean(yARC); nanmax(yARC)-nanmin(yARC); 0];
    
    opts = statset('nlinfit');
    opts.RobustWgtFun = 'bisquare';
    opts.MaxIter = 2000;
    
    p2mdl = fitnlm(xPRC,yPRC,cosMod,b0_2p); 
    a2mdl = fitnlm(xARC,yARC,cosMod,b0_2a);

    b3p=wrapTo2Pi(p2mdl.Coefficients.('Estimate')(3));
    b3a=wrapTo2Pi(a2mdl.Coefficients.('Estimate')(3));
    shift=wrapTo2Pi(b3p-b3a);
    
end