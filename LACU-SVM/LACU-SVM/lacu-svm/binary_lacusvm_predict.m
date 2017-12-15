function [prediction, accuracy, decision_value] = binary_lacusvm_predict(test_gnd, test_fea, model)

model0 = model.model0;

KTT = calc_kernel_matrix(test_fea', model.SVs_T',model.kernel_type, model.parameter);
KTU = calc_kernel_matrix(test_fea', model.SVs_U',model.kernel_type, model.parameter);

DTL = KTT(:,model.sv_indices_L) *  model.sv_coef_L;
DTU = KTU * model.sv_coef_U;

TA = mean(KTT,2);
DTA = TA * sum(model.sv_coef_AB(1:2));

DTN = KTT(:,model.sv_indices_L_neg) * model.sv_coef_AB(3:end);

decision_value0 = KTT(:,model0.sv_indices) * model0.sv_coef + model0.rho;
decision_value = DTL + DTU + DTA + DTN + model.rho;
pos_ind = (decision_value > 0);
decision_value(pos_ind) = decision_value0(pos_ind);
prediction = 2 * (decision_value > 0) - 1;
accuracy = sum(prediction == test_gnd) / length(test_gnd);