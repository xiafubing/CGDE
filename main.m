clc;
clear;
% 字典对+等距映射+因果保持
%% %数据读取
data_52 = load("TE数据\d00_te.dat");
data_33 = data_52(:,[1:22,42:52]);
data_33_mean = mean(data_33);
data_33_std = std(data_33);
data_source=(data_33-data_33_mean)./data_33_std;

% 分块操作
block1=[1,2,14,17,20,21,33];
block2 = [5,15,18,22,23,24,30];
block3 = [3,6,7,8,11,13,16,19,32];
block4 = [4,25,27];
block5 = [9,10,12,26,28,29,31];
data_block1 = data_source(:,block1)';
data_block2 = data_source(:,block2)';
data_block3 = data_source(:,block3)';
data_block4 = data_source(:,block4)';
data_block5 = data_source(:,block5)';
all_block_data = {data_block1,data_block2,data_block3,data_block4,data_block5};

%% %计算不同块的统计量与控制限
projections = {};
% admm迭代更新
% 初始化参数
alpha = 0.5;%     ts- 2.5     0.5   0.5
beta = 0.2;  %    ts-  5      -5    0.2 
gamma = 0.8; %    ts- -1      -1    0.8
lambda = 0.000001; % 0.00000001
stop_condition = 10^-6; % 35
Confidence = 0.01; 
dim=[3,3,6,2,1];
block_num=5;
total_trainingtime=0;
for i = 1:size(all_block_data,2)
    % 开始计时
    tic;
    disp(['第',num2str(i),'个子块']);
    projections{i}= admm(all_block_data{i},alpha,beta,gamma,lambda,stop_condition,i,dim(i));
    
    % 获取经过的时间
    trainingTime = toc;
    total_trainingtime=trainingTime+total_trainingtime;
    fprintf('训练时间: %.2f 秒 \n', trainingTime);
end

[p_incov,p_t2_lims,p_spe_lims]=compute_lims(projections,all_block_data,Confidence);

%% 循环所有故障
fault_num=[1,2,4,6,7,8,10,11,12,13,14,17,18,19,20];  % 15
all_t2=zeros(size(fault_num,2),1);
all_spe=zeros(size(fault_num,2),1);
all_error_t2=zeros(size(fault_num,2),1);
all_error_spe=zeros(size(fault_num,2),1);
all_fault_delay_t2 = zeros(size(fault_num,2),1);
all_fault_delay_spe = zeros(size(fault_num,2),1);
for j = 1: size(fault_num,2)
    f=fault_num(j);
    if f<10
        data_52_test = load(['TE数据\d0',num2str(f),'_te.dat']);
    else
        data_52_test = load(['TE数据\d',num2str(f),'_te.dat']);
    end
data_33_test = data_52_test(:,[1:22,42:52]);
data_test=(data_33_test-data_33_mean)./data_33_std;
data_block1_test = data_test(:,block1)';
data_block2_test = data_test(:,block2)';
data_block3_test = data_test(:,block3)';
data_block4_test = data_test(:,block4)';
data_block5_test = data_test(:,block5)';
X_test = {data_block1_test,data_block2_test,data_block3_test,data_block4_test,data_block5_test};
[P_T2_fused,P_SPE_fused] = compute_statics(projections,p_incov,p_t2_lims,p_spe_lims,X_test,Confidence);

alpha_lim = Confidence;
figure()
subplot(2,1,1);
plot(log10(P_T2_fused), 'b-'); hold on;
yline(log10(alpha_lim), 'r-');
ylabel('lg(BIC_{T2})', 'Interpreter', 'tex');
% SPE
subplot(2,1,2);
plot(log10(P_SPE_fused), 'b-'); hold on;
yline(log10(alpha_lim), 'r-');
xlabel('Sample Index'); ylabel('lg(BIC_{SPE})', 'Interpreter', 'tex');


% exportgraphics(gcf,'ouer_fault12.png', 'Resolution',300)
%% 故障检测结果统计
fault_start =161 ;
fault_detected_T2 = P_T2_fused(fault_start:end) > alpha_lim;
fault_detected_SPE = P_SPE_fused(fault_start:end) >alpha_lim;

index_t2 = find(fault_detected_T2 == 1, 1, 'first')+fault_start-1;  % 监测延迟
all_fault_delay_t2(j)=find(fault_detected_T2 == 1, 1, 'first')-1;
index_spe = find(fault_detected_SPE == 1, 1, 'first')+fault_start-1;
all_fault_delay_spe(j)=find(fault_detected_SPE == 1, 1, 'first')-1;
% exportgraphics(gcf,['te_our_fault',num2str(j),'first_fault_indx_t2_',num2str(index_t2),'_spe_',num2str(index_spe),'.png'], 'Resolution',300)


fprintf('故障：%d\n', f);
fprintf('故障检测统计（从样本%d开始）：\n', fault_start);
fprintf('T^2 检测率: %.2f%%\n', mean(fault_detected_T2)*100);
fprintf('SPE 检测率: %.2f%%\n', mean(fault_detected_SPE)*100);
% 
fault_error_detected_T2 = P_T2_fused(1:(fault_start-1)) > alpha_lim;
fault_error_detected_SPE = P_SPE_fused(1:(fault_start-1)) >alpha_lim;
fprintf('T^2 误报率: %.2f%%\n', mean(fault_error_detected_T2)*100);
fprintf('SPE 误报率: %.2f%%\n', mean(fault_error_detected_SPE)*100);

all_t2(j)=mean(fault_detected_T2);
all_spe(j)=mean(fault_detected_SPE);
all_error_t2(j)=mean(fault_error_detected_T2);
all_error_spe(j)=mean(fault_error_detected_SPE);
end

fprintf('所有故障平均\n');
fprintf('T^2 检测率: %.2f%%\n', mean(all_t2)*100);
fprintf('SPE 检测率: %.2f%%\n', mean(all_spe)*100);
fprintf('T^2 误报率: %.2f%%\n', mean(all_error_t2)*100);
fprintf('SPE 误报率: %.2f%%\n', mean(all_error_spe)*100);
disp(['T^2 监测延迟: ',num2str(mean(all_fault_delay_t2))])
disp(['SPE 监测延迟: ',num2str(mean(all_fault_delay_spe))])



