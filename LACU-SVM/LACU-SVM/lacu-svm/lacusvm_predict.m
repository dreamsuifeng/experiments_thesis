function [predictions, Acc, MacroF1] = lacusvm_predict(label_vector, instance_matrix, aug_class_id, model)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lacusvm implements the LACU-SVM algorithm as shown in [1].
%
%    lacusvm employs the svqp2 SMO sovler in [2] 
% 
%    Syntax
%
%       [predctions, MacroF1] = lacusvm_predict(y, x, model)
%
%    Description
%
%       lacusvm_predict takes,
%       label_vector    - A Nx1 test label vector, where N is the number of
%                         test instances, and y_i \in \{1,2,...,K\}
%     instance_matrix   - A Nxd test instances matrix, where L is the number of
%                         test instances, and d is the number of attributes
%       aug_class_id    - ID of augmented classes
%         model         - A trained model;
%
%      and returns,
%       predctions      - Prediction vector
%          Acc          - Accuracy
%        MacroF1        - MacroF1
%
% [1] Qing Da, Yang Yu, and Zhi-Hua Zhou. Learning with Augmented Class by
%     Exploiting Unlabeled Data. In: Proceedings of the 28th AAAI Conference
%     on Artificial Intelligence (AAAI'14), Qu¨¦bec city, Canada, 2014.
% [2] Bottou, L., and Lin, C.-J. 2007. Support vector machine
%     solvers. Large Scale Kernel Machines, 301-320.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labelSet = model.labelSet;
labelSetSize = length(labelSet);
models = model.models;
decv= zeros(size(label_vector, 1), labelSetSize);

for i=1:labelSetSize
  [~, ~, decision_value] = binary_lacusvm_predict(2 * (label_vector == labelSet(i)) - 1, instance_matrix, models{i});
  decv(:, i) = decision_value;
end

[max_val, ind] = max(decv, [], 2);
predictions = labelSet(ind);
predictions(max_val < 0) = aug_class_id;
[Acc, MacroF1] = evaluate(label_vector, predictions);

end

function [Acc, MacroF1] = evaluate(labels, predictions)

Acc = sum(predictions == labels) / length(labels);

label_set = unique(labels);
num_class = length(label_set);

F1s = zeros(num_class, 1);

for i=1:num_class
    index1 = labels == label_set(i);
    index2 = predictions == label_set(i); 
    A = sum(index1 & index2);
    
    if sum(index2) ~= 0
        prec = A / sum(index2);
        rec = A / sum(index1);
    
        if prec + rec ~= 0
            F1s(i) = 2*prec*rec / (prec + rec);
        end
    end
end

MacroF1 = mean(F1s(1:end));

end
