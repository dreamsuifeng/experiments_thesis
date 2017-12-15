addpath('util');
addpath('DMNL');
clear;
load dt/MSCV2;
 s = RandStream.create('mt19937ar','seed',1);
        RandStream.setGlobalStream(s);
 
%maximum iteration number
param.max_iter = 500;
%maximum inner iteration number for W & H
param.inner_iter = 50;
%regularizer parameters for H and W respectively
param.lambda1 = 0.00001;
param.lambdaw = 0.00001;
%the number of potential new labels
param.k = 4;
%ADMM parameter
param.rho = 0.125;
%Other gradient descent parameters
param.step_size = .03125/8;
param.clipnorm = 10000;
param.decayw = 0;
param.decayh = 0;
param.decayhsquare = 1;
param.disc = 0.2;

Bags_back = Bags;
for i = 1:length(Bags)
     Bags{i} = Bags{i}';
end

X = Bags;
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


for rr = 1:1

  

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


for i= 1:length(X)
    yi = Y(i,:);
    Bag = X{i};
    bgsz = size(Bag,1);
    
    y_i = y{i};
    y{i} = y_i;
    Bags{i} = Bag;
end



train_data = Bags(trnidx{1});
train_target = Y(trnidx{1},1:19);
train_inst_label = y(trnidx{1});

[W, AW, Anum, H]=dmnl_train(train_target, train_data, param);
Wm = 0;
for i = 1:length(AW)
     Wm = Wm + AW{i};
end
Wm = Wm/length(AW);

end

%Instance Annotation Accuracy
Fp = get_g_func(cell2mat(Bags)*(Wm));
[~,idx]  = max(Fp,[],2);
gt = cell2mat(y);
[~,idx2]  = max(gt,[],2);
acc = sum(idx==idx2)/length(idx)

%New label detection AUC
 Fn = max(Fp(:,end-param.k+1:end),[],2);
  nl = idx2>=20;
  [~,~,~,AUC] = perfcurve(nl,Fn,1)
