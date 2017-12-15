addpath('util');
addpath('DMNL');
clear;
load dt/sydata;
 s = RandStream.create('mt19937ar','seed',1);
        RandStream.setGlobalStream(s);

%maximum iteration number
param.max_iter = 10;
%maximum inner iteration number for W & H
param.inner_iter = 50;
%regularizer parameters for H and W respectively
param.lambda1 = 0.00001;
param.lambdaw = 0.00001;
%the number of potential new labels
param.k = 2;
%ADMM parameter
param.rho = 0.125;
%Other gradient descent parameters
param.step_size = .03125/8;
param.clipnorm = 10000;
param.decayw = 0;
param.decayh = 0;
param.decayhsquare = 1;
param.disc = 0.2;

Bags = X';
Bags_back = Bags;


[X, ~]=AddBoundBag(X,0,20,100);
X_bound_data=X{length(X)};
X_data=X(1:length(X)-1); 

k=200; 
g=randn(k,size(X{1},1)); 
save('g.mat','g');

[X]=PreprocessingX(X,'kernel',[],0.1); 

for i=1:length(Bags)
    z = y{i};
    y{i} = z;
    X{i} = X{i}';
    Bags_back{i} = Bags_back{i}';
end
X{end} = X{end}';

for rr = 1:1;

  

insts = cell2mat(X);
zz = sum(insts.^2,1);

mx = max(insts,[],1);
mn = min(insts,[],1);

Y(Y<0) = 0;


for i= 1:length(X)
   
    Bag = X{i};
    bgsz = size(Bag,1);
   
    Bag = (Bag-repmat(mn,bgsz,1))./(repmat(mx-mn,bgsz,1)+1e-6);

    X{i} = Bag;

end


for i= 1:length(X)-1
    yi = Y(i,:);
    Bag = X{i};
    bgsz = size(Bag,1);
    
    y_i = y{i};
    y{i} = y_i;
    Bags{i} = Bag;
end



train_data = Bags;
train_target = Y(:,1:4);

[W, AW, Anum, H]=dmnl_train(train_target, train_data, param);
Wm = 0;
for i = 1:length(AW)
     Wm = Wm + AW{i};
end
Wm = Wm/length(AW);


insts = X{end};
F = get_g_func(insts*(Wm));
figure;
PlotResultBoundary(F,X_bound_data,19, 1);

end

Fp = get_g_func(cell2mat(Bags)*(Wm));
[~,idx]  = max(Fp,[],2);
gt = cell2mat(y');
[~,idx2]  = max(gt,[],2);
%%instance annotation accuracy.
acc = sum(idx==idx2)/length(idx)

