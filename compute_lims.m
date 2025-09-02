
% 离线建模-计算统计量
function [p_incov,p_t2_lims,p_spe_lims] = compute_lims(projections,X,alpha1)
n_test = size(X{1},2);
n_blocks = size(X,2);
[p_t2_lims,p_spe_lims ]=deal(zeros(n_blocks,1));
alpha=alpha1;
p_incov={};
for b = 1:n_blocks
    Xb_test = X{b};
    P = projections{b};
    incov = P*Xb_test*Xb_test'*P'/(size(Xb_test,2)-1);
    incov =incov +1e-6 * eye(size(incov));
    p_incov{b}= incov;
    I = eye(size(P,2));
    for i = 1:n_test
        T2_stats(i,b) = Xb_test(:,i)'*P'*inv(incov)*P*Xb_test(:,i);
        SPE_stats(i,b) =sum(((I - P'*P)*Xb_test(:,i)).^2);
    end
    [f, xi] = ksdensity(T2_stats(:,b));
    cdf = cumsum(f) * (xi(2) - xi(1)); 
    p_t2_lims(b) = xi(find(cdf >= (1-alpha), 1));
    [f, xi] = ksdensity(SPE_stats(:,b));
    cdf = cumsum(f) * (xi(2) - xi(1)); 
    p_spe_lims(b) = xi(find(cdf >= (1-alpha), 1));
end
end



