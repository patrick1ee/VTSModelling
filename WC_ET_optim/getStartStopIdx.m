function [start,stop]=getStartStopIdx(x)

    % used in block method to get start/stop idx of pulses

    n=length(x);
    
    start=x(1);
    
    if x(2)~=x(1)+1
        stop=x(1);
    else
        stop=[];
    end
    
    for i=2:n-1
        if x(i)~=x(i-1)+1
            start=[start;x(i)];
        end
                  
        if x(i+1)~=x(i)+1
            stop=[stop;x(i)];
        end
    end
    
    stop=[stop;x(n)];
    
    if x(n)~=x(n-1)+1
        start=[start;x(n)];
    end
        
end


        
        