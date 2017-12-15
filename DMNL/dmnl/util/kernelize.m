function [X_out]=kernelize(X, kerneltype, width)
%%Compute random Fourier feature, which is equivalent to a kernel mapping
load('g.mat'); 
g = g*width;

for i = 1:length(X)
    X_out{i}=[cos(X{i}'*g'), sin(X{i}'*g')]';
end
X_out = X_out';
end
