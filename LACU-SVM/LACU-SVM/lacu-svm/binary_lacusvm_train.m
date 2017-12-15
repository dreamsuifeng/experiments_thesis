function model = binary_lacusvm_train(train_gnd, train_fea, unlabel_fea, C, C_star, ramp_s, eta, lambda, max_iter, kernel_type, parameter, verbosity)

mean_Y = mean(train_gnd);

[L, dim] = size(train_fea);
[U, ~] = size(unlabel_fea);

if isempty(parameter)
    switch kernel_type
        case 0
            parameter = [];
        case 1
            parameter = [1/dim 0 3];
        case 2
            parameter = 1/dim;
    end
end

ind_neg = train_gnd == -1;
NN = sum(ind_neg);

n = L + 2 * U + 2 + NN;

%% calculate kernel
K = zeros(n, n);

KL = calc_kernel_matrix(train_fea',train_fea',kernel_type, parameter);
KU = calc_kernel_matrix(unlabel_fea',unlabel_fea',kernel_type, parameter);
KLU = calc_kernel_matrix(train_fea',unlabel_fea',kernel_type, parameter);

KLA = mean(KLU,2);
KUA = mean(KU,2);
KAA = mean(KU(:));

KLB = KL(:,ind_neg);
KUB = KLU(ind_neg,:)';
KAB = KLA(ind_neg)';
KBB = KL(ind_neg,ind_neg);

K(1:L, 1:L) = KL;
K(1:L, L+1:L+2*U) = [KLU KLU];
K(L+1:L+2*U, 1:L) = K(1:L, L+1:L+2*U)';
K(L+1:L+2*U, L+1:L+2*U) = [KU KU;KU KU];

K(1:L+2*U, L+2*U+1:L+2*U+2) = [KLA KLA;KUA KUA;KUA KUA];
K(L+2*U+1:L+2*U+2, 1:L+2*U) = K(1:L+2*U, L+2*U+1:L+2*U+2)';
K(L+2*U+1:L+2*U+2, L+2*U+1:L+2*U+2) = [KAA KAA;KAA KAA];

K(1:L+2*U+2, L+2*U+3:end) = [KLB;KUB;KUB;KAB;KAB];
K(L+2*U+3:end, 1:L+2*U+2) = K(1:L+2*U+2, L+2*U+3:end)';
K(L+2*U+3:end, L+2*U+3:end) = KBB;

clear KL KU KLU KLA KUA KAA KLB KUB KAB KBB;

Y = zeros(n, 1);
Y(1:L) = train_gnd;
Y(L+1:L+U) = 1;
Y(L+U+1:L+2*U) = -1;
Y(L+2*U+1) = -1;
Y(L+2*U+2) = 1;
Y(L+2*U+3:end) = -1;


%% train basic svm using labeled data only
b = train_gnd;

cmin0 = zeros(L ,1);
cmax0 = zeros(L ,1);

index = train_gnd == 1;
cmin0(index) = 0;
cmax0(index) = C;

index = train_gnd == -1;
cmin0(index) = -C;
cmax0(index) = 0;


[alpha0, b0, fx] = svqp2mex(K(1:L, 1:L), b, cmin0, cmax0, verbosity);

model0.kernel_type = kernel_type;
model0.parameter = parameter;
model0.sv_indices = find(alpha0 ~= 0);
model0.totalSV = length(model0.sv_indices);
model0.rho = b0;
model0.sv_coef = alpha0(model0.sv_indices);
model0.SVs = train_fea(model0.sv_indices,:);

%% solve optimization by CCCP
neg_inf = -1e30;
pos_inf = 1e30;

beta = zeros(n,1);
b = zeros(n, 1);

b(1:L+2*U) = Y(1:L+2*U);
b(L+2*U+1) = mean_Y;
b(L+2*U+2) = max(-0.9,eta * mean_Y);

cmin = zeros(n ,1);
cmax = zeros(n ,1);

cmin(1:L) = cmin0;
cmax(1:L) = cmax0;

iter = 1;
while iter <= max_iter
    
    if iter == 1
        [~, ~, dec_U] = svqp2predict(zeros(U,1), unlabel_fea, model0);
    end
    
    tmp = fx(~ind_neg);
    fxz = max(0,min(tmp));
    
    if fxz == 0
        fxz = fxz + 0.5;
    else
        fxz = fxz + lambda;
    end
    
    b(L+2*U+3:end) = -fxz;
    
    if iter == 1
        yfx_U = Y(L+1:L+2*U) .* [dec_U;dec_U];
    else
        yfx = Y .* fx;
        yfx_U = yfx(L+1:L+2*U);
    end
    beta(L+1:L+2*U) = C_star * (yfx_U < ramp_s);
    
    cmin(L+1:L+U) = -beta(L+1:L+U);
    cmax(L+1:L+U) = C_star - beta(L+1:L+U);
    
    cmin(L+U:L+2*U) = -C_star+beta(L+U:L+2*U);
    cmax(L+U:L+2*U) = beta(L+U:L+2*U);
    
    cmin(L+2*U+1) = neg_inf;
    cmax(L+2*U+1) = 0;
    
    cmin(L+2*U+2) = 0;
    cmax(L+2*U+2) = pos_inf;
    
    cmin(L+2*U+3:end) = neg_inf;
    cmax(L+2*U+3:end) = 0;
    
    [alpha, b0, fx] = svqp2mex(K, b, cmin, cmax, verbosity);
    
    iter = iter + 1;
end

%% save to model
model.model0 = model0;

model.kernel_type = kernel_type;
model.parameter = parameter;

model.sv_indices_L_neg = ind_neg;

alpha_L = alpha(1:L);
model.sv_indices_L = alpha_L ~= 0;
model.sv_coef_L = alpha_L(model.sv_indices_L);
model.SVs_T = train_fea;

alpha_U = alpha(L+1:L+2*U);
model.sv_indices_U = alpha_U ~= 0;
model.sv_coef_U = alpha_U(model.sv_indices_U);
model.SVs_U = [unlabel_fea(model.sv_indices_U(1:U),:);unlabel_fea(model.sv_indices_U(U+1:2*U),:)];

model.sv_coef_AB =  alpha(L+2*U+1:end);

model.rho = b0;

end

function [prediction, accuracy, decision_value]=svqp2predict(test_gnd, test_fea, model)
K_test = calc_kernel_matrix(test_fea',model.SVs',model.kernel_type, model.parameter);
decision_value = K_test * model.sv_coef + model.rho;
prediction = 2 * (decision_value > 0) - 1;
accuracy = sum(prediction == test_gnd) / length(test_gnd);

end
