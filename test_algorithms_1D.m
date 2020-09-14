clc; close all;clear all;
addpath(genpath(pwd));
%% 
% ��������ʵ��A��Ѱ�����Ż��ĵ�



% problem 1:����x������ֵ������
%2����ֵ��ѡ��
% 2.sinc conv bernoulli ind =20 ,�ɽ�
% Comparing the algorithmic performance of the proposed nonconvex
% optimization methods in the paper
% ``Short-and-Sparse Deconvolution -- A Geometric Approach''
% Yenson Lau*, Qing Qu*, Han-Wen Kuo, Pengcheng Zhou, Yuqian Zhang, and John Wright
% (* denote equal contribution)
%
% We solve the short-and-sparse convolutional dictionary learning problem
% y = sum_{k=1}^K a0k conv x0k + b * 1 + n
%with both a0k and x0k unknown, b is a constant bias, n is noise
%
% The algorithms solve the following 1D optimization problem
% min F(A,X) = 0.5 * ||y - sum_{k=1}^K ak conv xk||_2^2 + lambda * ||X||_1
% s.t. ||ak|| = 1, k = 1,...,K
% A = [a1,a2,...,aK], X = [x1,x2,...,xK]
%
% Test the proposed Alternating desecent method (ADM), inertial ADM (iADM),
% homotopy acceleration and reweighting method
% Code written by Qing Qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% platform for simulation of Convolutionoal dictionary learning problem
%
%% optimization parameters
opts.tol = 1e-6; % convergence tolerance ����ϵ��
opts.isnonnegative = false; % enforcingǿ�� nonnegativity�Ǹ��� on X
opts.isupperbound = false; % enforce upper bound �Ͻ�on X
opts.upperbound = 1.5; % upper bound number �Ͻ���
opts.hard_thres = false; % hard-thresholdӲ��ֵ on small entries of X to zero
opts.MaxIter = 1000; % number of maximum iterations ����������,ԭʼ������������Ϊ1000
opts.MaxIter_reweight = 10; % reweighting iterations for reweighting algorithm reweighting �㷨��������
opts.isbias = true; % enforce when there is a constant bias in y 
opts.t_linesearch = 'bt'; % linesearch �������� for the stepsize t for X
opts.err_truth = true; % enforce to compute error w.r.t. the groundtruth for (a0, x0)
opts.isprint = true; % print the intermediate�м�� result


%% generate the measurements

% setup the parameters
n = 200; % length of each kernel a0k ��a0�ĳ���
m = 1000; % length of the measurements y   ���y�ĳ���
K = 1; % number of kernels �˵�����
% theta = n^(-4/5); % sparsity parameter for Bernoulli distribution ��Ŭ���ֲ�ϡ��ϵ��
theta = n^(-1);
opts.lambda = 1e-2; % penalty parameter lambda


a_type = 'echo'; % choose from 'randn', 'ar1', 'ar2', 'gaussian', 'sinc'
x_type = 'bernoulli-gaussian'; % choose 'bernoulli' or
% 'bernoulli-rademacher' or 'bernoulli-gaussian'����twopulse��
b_0 = 1; % bias
noise_level = 0;% noise level������

% generate the data ����ģ������
% y0 = a_0 conv b_0 + bias 
% y = a_0 conv b_0 + bias + noise ��noise��Ϊ0����Ŀǰy0��y��ͬ
[A_0, X_0, y_0, y] = gen_data( theta, m, n, b_0, noise_level, a_type, x_type);
opts.truth = true;
opts.A_0 = A_0; opts.X_0 = X_0; opts.b_0 = b_0;
y_0 = y;



%% initialization for A, X, b

% initialize A
opts.A_init = zeros(3*n,K);
% opts.A_init = zeros(n,K);

% L = length(y_0);
%     window = n;
%     for i = 1 : L-window
%         total_E = 0;
%         for nn = i : i + window
%             E{nn}=y_0(nn)*y_0(nn);
%             total_E = total_E + E{nn};
%         end
%         E_win{i} = total_E;
%     end
%     max = 0 ;
%     max_i = 0 ;
%     for i = 1 : L-window
% 
%         if (E_win{i}>=max)
%             max = E_win{i};
%             max_i = i;
%          else
%             max = max;
%             max_i = max_i;  
%         end
%       
%     end
%     ind = max_i;
r_result = [];

  L = length(y_0);
    window = n;
    for i = 1 : L-window
        total_E = 0;
        for nn = i : i + window
            E{nn}=y_0(nn)*y_0(nn);
            total_E = total_E + E{nn};
        end
        E_win{i} = total_E;
    end
    max = 0 ;
    max_i = 0 ;
    for i = 1 : L-window

        if (E_win{i}>=max)
            max = E_win{i};
            max_i = i;
         else
            max = max;
            max_i = max_i;  
        end
      
    end
H_pianyi = max_i;
for ind = max_i :max_i + 100
% for ind = 1:3
for k = 1:K
%     ind = randperm(m,1);
%    ind = 20;%ֻ����ʱ������ĳ�ʼ������ ind = 74ʱ�����Ϊ���ã�60
%     ind = 1;
%     while  y(ind)<=mean(y)
%         ind = ind + 1;
%     end
%   L = length(y_0);
%     window = n;
%     for i = 1 : L-window
%         total_E = 0;
%         for nn = i : i + window
%             E{nn}=y_0(nn)*y_0(nn);
%             total_E = total_E + E{nn};
%         end
%         E_win{i} = total_E;
%     end
%     max = 0 ;
%     max_i = 0 ;
%     for i = 1 : L-window
% 
%         if (E_win{i}>=max)
%             max = E_win{i};
%             max_i = i;
%          else
%             max = max;
%             max_i = max_i;  
%         end
%       
%     end
%     ind = max_i;
%     ind = max_i - window/2;
    y_pad = [y_0;y_0];%y_pad ��ĩ���㣿
    a_init = y_pad(ind:ind+n-1);  %a�ĳ�ʼ������ԭ��
%     a_init = [ zeros(n,1);a_init; zeros(n,1)];
    a_init = [a_init; zeros(n,1); zeros(n,1)];% 7��14���޸�A�ĳ�ʼ����ԭʼ����Ϊ��a_init = [ zeros(n,1);a_init; zeros(n,1)];
%     a_init = [A_0; A_0; A_0];
    a_init = a_init / norm(a_init);
    opts.A_init(:,k) = a_init;
   
end

opts.X_init = zeros(m,K); % initialize X
opts.b_init = mean(y);  %b��ʼ��Ϊy�ľ�ֵ
opts.W = ones(m,K); % initialize the weight matrix

%% run the optimization algorithms
Alg_num = 4;

% Alg_type = {'ADM','iADM','homotopy-ADM','homotopy-iADM','reweighting'};
Alg_type = {'iadm'};
Psi_min = Inf; psi_min = Inf;
Psi = cell(length(Alg_type),1);
psi = cell(length(Alg_type),1);
Err_A = cell(length(Alg_type),1);

for k = 1:length(Alg_type)
    
    switch lower(Alg_type{k})
        case 'adm'
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = ADM( y_0, opts);
        case 'iadm'
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = iADM( y_0, opts);
        case 'homotopy-adm'
            opts.homo_alg = 'adm';
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = homotopy( y_0, opts);
        case 'homotopy-iadm'
            opts.homo_alg = 'iadm';
            % [A, X, b, Psi_Val, psi_Val, Err_A, Err_X]= homotopy(y_0, opts)
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = homotopy( y_0, opts);
        case 'reweighting'
            opts.reweight_alg = 'iadm';
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = reweighting( y_0, opts);
    end
    
end

L = length(A);
    window = 100;
    for i = 1 : L-window
        total_E = 0;
        for nn = i : i + window
            E{nn}=A(nn)*A(nn);
            total_E = total_E + E{nn};
        end
        E_win{i} = total_E;
    end
    max = 0 ;
    max_i = 0 ;
    for i = 1 : L-window

        if (E_win{i}>=max)
            max = E_win{i};
            max_i = i;
         else
            max = max;
            max_i = max_i;  
        end
      
    end
    indA = max_i;

r = corr2(A_0,A(indA :indA+99));
% ��r�ӽ�-1ʱ�����Կ��Ƕ�A��90�����Ƶ���
if r > 0.99
r_result = [r_result ind];
end 
str = ['ind ȡֵΪ',num2str(ind),'A�����ƶ�Ϊ',num2str(r)];
figure('name',str)


subplot(2,1,1)
plot(A_0)
hold on
plot(A(indA :indA+100) )
hold off
subplot(2,1,2)
plot(A)
hold off
legend
ind = ind + 1;
end
% figure()
% plot(y)
% hold on
% plot(r_result)
% hold off
%% ����һ����ȷ������˴�����ΪѰ�Һ��ʵĳ�ʼֵ
% total_right = find(r_result>0.8);
ind = r_result(1,1);
for k = 1:K

    y_pad = [y_0;y_0];%y_pad ��ĩ���㣿
    a_init = y_pad(ind:ind+n-1);  %a�ĳ�ʼ������ԭ��
%     a_init = [ zeros(n,1);a_init; zeros(n,1)];
    a_init = [a_init; zeros(n,1); zeros(n,1)];% 7��14���޸�A�ĳ�ʼ����ԭʼ����Ϊ��a_init = [ zeros(n,1);a_init; zeros(n,1)];
%     a_init = [A_0; A_0; A_0];
    a_init = a_init / norm(a_init);
    opts.A_init(:,k) = a_init;
   
end

opts.X_init = zeros(m,K); % initialize X
opts.b_init = mean(y);  %b��ʼ��Ϊy�ľ�ֵ
opts.W = ones(m,K); % initialize the weight matrix

%% run the optimization algorithms
Alg_num = 4;

% Alg_type = {'ADM','iADM','homotopy-ADM','homotopy-iADM','reweighting'};
Alg_type = {'iadm'};
Psi_min = Inf; psi_min = Inf;
Psi = cell(length(Alg_type),1);
psi = cell(length(Alg_type),1);
Err_A = cell(length(Alg_type),1);

for k = 1:length(Alg_type)
    
    switch lower(Alg_type{k})
        case 'adm'
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = ADM( y_0, opts);
        case 'iadm'
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = iADM( y_0, opts);
        case 'homotopy-adm'
            opts.homo_alg = 'adm';
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = homotopy( y_0, opts);
        case 'homotopy-iadm'
            opts.homo_alg = 'iadm';
            % [A, X, b, Psi_Val, psi_Val, Err_A, Err_X]= homotopy(y_0, opts)
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = homotopy( y_0, opts);
        case 'reweighting'
            opts.reweight_alg = 'iadm';
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = reweighting( y_0, opts);
    end
    
  
    
end



% figure('name','���Ϊ')
% subplot(2,1,1)
% plot(A)
% hold on
% subplot(2,1,2)
% plot(y_0)
% hold on
% plot(X,'b')
% hold on 
% plot(X_0,'r')
% hold off
% legend
%% ���Ͽ��Եõ���ȷ��A��������A�õ���ȷ��X


 L = length(A);
    window = 100;
    for i = 1 : L-window
        total_E = 0;
        for nn = i : i + window
            E{nn}=A(nn)*A(nn);
            total_E = total_E + E{nn};
        end
        E_win{i} = total_E;
    end
    max = 0 ;
    max_i = 0 ;
    for i = 1 : L-window

        if (E_win{i}>=max)
            max = E_win{i};
            max_i = i;
         else
            max = max;
            max_i = max_i;  
        end
      
    end
    ind = max_i;
% for ind = max_i :max_i+200
% figure(1)
% plot(h)
% hold on
% plot(A(ind : ind + 199)*5)
% hold off
% h_toep = 5*A(ind : ind + 199).';
% r = [h_toep zeros(1,length(y_0)-1)];
% c = [h_toep(1) zeros(1,length(y_0)-1)];
% hConv = toeplitz(r,c);
% 
% [x_slove,x_debias,objective,times,debias_start,mses,max_svd] = ...
%          TwIST(y_0,hConv,0.5, 'lambda', 1e-4, 'StopCriterion', 3, 'ToleranceA', 0.0001);
%      
%      
% figure('name','final result')
% plot(A_0)
% hold on
% plot(A(ind : ind + 99))
% hold off
% 
% figure()
% plot(y)
% hold on
% plot(x_slove,'r')
% plot(X_0)
% hold off


