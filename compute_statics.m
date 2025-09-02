% 在线监控，计算统计量
function [P_T2_fused,P_SPE_fused] = compute_statics(projections,p_incov,p_t2_lims,p_spe_lims,X_test,Confidence)
n_test = size(X_test{1},2);
n_blocks = size(X_test,2);
T2_stats = zeros(n_test, n_blocks);
SPE_stats = zeros(n_test, n_blocks);
alpha=Confidence;
for b = 1:n_blocks
    Xb_test = X_test{b};
    P = projections{b};
    incov =p_incov{b} ;
    I = eye(size(P,2));   
    for i = 1:n_test
        T2_stats(i,b) = Xb_test(:,i)'*P'*inv(incov)*P*Xb_test(:,i);
        SPE_stats(i,b) = sum(((I - P'*P)*Xb_test(:,i)).^2);
    end
end

[P_T2, P_SPE] = deal(zeros(n_test, n_blocks));
[P_T2_fault, P_SPE_fault] = deal(zeros(n_test, n_blocks));

for b = 1:n_blocks
    
    P_T2_normal(:,b) = exp(-T2_stats(:,b) /  p_t2_lims(b));
    P_T2_fault(:,b) = exp(-p_t2_lims(b) ./ T2_stats(:,b));
    
    P_SPE_normal(:,b) = exp(-SPE_stats(:,b) /  p_spe_lims(b));
    P_SPE_fault(:,b) = exp(-p_spe_lims(b) ./ SPE_stats(:,b));
    
    P_T2(:,b) = (P_T2_fault(:,b) * alpha) ./ (P_T2_normal(:,b)*(1-alpha) + P_T2_fault(:,b)*alpha);
    P_SPE(:,b) = (P_SPE_fault(:,b) * alpha) ./ (P_SPE_normal(:,b)*(1-alpha) + P_SPE_fault(:,b)*alpha);
end
% writematrix(P_SPE,"fault7_te_P_spe.xlsx");
% writematrix(P_T2,"fault7_te_P_t2.xlsx");
weights_T2 = P_T2_fault ./ sum(P_T2_fault, 2);
weights_SPE = P_SPE_fault ./ sum(P_SPE_fault, 2);
P_T2_fused = sum(P_T2 .* weights_T2, 2);
P_SPE_fused = sum(P_SPE .* weights_SPE, 2);
end