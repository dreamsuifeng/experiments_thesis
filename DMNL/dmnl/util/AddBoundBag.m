function [X, X_bound]=AddBoundBag(X,minx,maxx,resolution)
u = linspace(minx,maxx, resolution);
v = linspace(minx, maxx, resolution);
X_bound=[];
for i = 1:length(u)
    for j = 1:length(v)
        X_bound=[X_bound [u(i) v(j)]'];
    end
end
X{length(X)+1}=X_bound;
end