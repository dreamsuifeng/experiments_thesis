function model = lacusvm_train(label_vector, instance_matrix, u_instance_matrix, C, C_star, ramp_s, eta,...
                        lambda, max_iter, kernel_type, kernel_parameter, verbosity)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lacusvm implements the LACU-SVM algorithm as shown in [1].
%
%    lacusvm employs the svqp2 SMO sovler in [2] 
% 
%    Syntax
%
%       model = lacusvm_train(y, x, unlabel_fea, C, C_star, ramp_s, eta,
%                   lambda, max_iter, kernel_type, parameter, verbosity)
%
%    Description
%
%       lacusvm_train takes,
%      label_vector     - A Lx1 training label vector, where L is the number of
%                         training instances, and y_i \in \{1,2,...,K\}
%     instance_matrix   - A Lxd training instances matrix, where L is the number of
%                         training instances, and d is the number of attributes
%    u_instance_matrix  - A Uxd unlabeled instances matrix, where U is the
%                         numebr of unlabeled instances
%        C,C_star       - Regularization parameters
%         ramp_s        - Parameter in ramp loss, needs to be in (-1,0]
%          eta          - Parameter which controls the fraction of positive
%                         instance
%         lambda        - Parameter which controls the move if seperator
%                         surporting a decision boundary closer to labeled examples
%         max_iter      - Maximal number of iterations
%        kernel_type    - Type of kernels, 0/1/2 for linear/poly/gaussian
%    kernel_parameter   - Parameter for kernels ([] for linear, [gamma coef degree] for
%                         poly, [gamam] for gaussian)
%       verbosity       - Verbosity (0 for nothing, 1 for details)
%
%      and returns,
%         model         - A trained model;
%
% [1] Qing Da, Yang Yu, and Zhi-Hua Zhou. Learning with Augmented Class by
%     Exploiting Unlabeled Data. In: Proceedings of the 28th AAAI Conference
%     on Artificial Intelligence (AAAI'14), Qu¨¦bec city, Canada, 2014.
% [2] Bottou, L., and Lin, C.-J. 2007. Support vector machine
%     solvers. Large Scale Kernel Machines, 301-320.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labelSet = unique(label_vector);
labelSetSize = length(labelSet);
models = cell(labelSetSize,1);

for i=1:labelSetSize
    fprintf('training for class %d\n',labelSet(i));
    models{i} = binary_lacusvm_train(2 * (label_vector == labelSet(i)) - 1, instance_matrix, u_instance_matrix,...
        C, C_star, ramp_s, eta, lambda, max_iter, kernel_type, kernel_parameter, verbosity);
end

model = struct('models', {models}, 'labelSet', labelSet);
