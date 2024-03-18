function [ out ] = avgAngle( in )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% in: vector of angles in rad
% out: angle of the sum of unit vectors in rad

N=length(in);
x=0;
y=0;

for i=1:N
    x=x+cos(in(i));
    y=y+sin(in(i));
end

out=atan2(y,x);

end

