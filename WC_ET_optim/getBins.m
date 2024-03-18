function [ edges,barycenters ] = getBins( dataVect, nBins )
%binning across first dimension
%dataVect should be the values to be binned

pVect=0:100/nBins:100;

edges=zeros(1,nBins);


for p=1:length(pVect)
    pVal=pVect(p);
    edges(p)= prctile(dataVect,pVal);
end

edges(1)=min(0,edges(1));


barycenters=zeros(1,nBins-1);

for p=1:length(pVect)-1
    if p==length(pVect)-1
        idx=find(dataVect<=edges(p+1) & dataVect>=edges(p));
    else
        idx=find(dataVect<edges(p+1) & dataVect>=edges(p));
    end
    
%     if isempty(idx)
%         keyboard
%     end
    
    barycenters(p)=mean(dataVect(idx));
end


end

