clear;

% add path 
addpath('lacu-svm/svqp2mex');
addpath('lacu-svm');

% laod data
load('mnist_demo.mat');

% use linear kernel with C = 1
C = 1;
kernel_type = 0;

% use default setting
C_star = C * size(instance_train_matrix,1) / size(u_instance_matrix,1);
ramp_s = -0.3;
eta = 1.3;
lambda = 0.1;
max_iter = 10;

% train
model = lacusvm_train(label_train_vector, instance_train_matrix, u_instance_matrix, C, C_star,...
    ramp_s, eta, lambda, max_iter, kernel_type, [], 1);

% test
[predctions, Acc, MacroF1] = lacusvm_predict(label_test_vector, instance_test_matrix, aug_class_id, model);

% print accuracy and macro-F1
fprintf('Acc = %.4f, MacroF1 = %.4f\n', Acc, MacroF1); % shoud be "Acc = 0.8450, MacroF1 = 0.8345" 
