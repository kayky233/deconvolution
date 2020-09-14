% n ��a0�ĳ��ȣ�m y�ĳ���
function [A, X] = deconvolutionmodel(y,n,m)

addpath(genpath(pwd));
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

K = 1; % number of kernels �˵�����
opts.lambda = 1e-2; % penalty parameter lambda
y_0 = y;
%% initialization for A, X, b

% initialize A
opts.A_init = zeros(3*n,K);
% opts.A_init = zeros(n,K);
for k = 1:K
%     ind = randperm(m,1);
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
    ind = max_i + + randperm(10,1);
    y_pad = [y_0;y_0];%y_pad ��ĩ���㣿
    a_init = y_pad(ind:ind+n-1);  %a�ĳ�ʼ������ԭ��
%     a_init = [ zeros(n,1);a_init; zeros(n,1)];
     a_init = [a_init; zeros(n,1); zeros(n,1)];% 7��14���޸�A�ĳ�ʼ����ԭʼ����Ϊ��a_init = [ zeros(n,1);a_init; zeros(n,1)];
    a_init = a_init / norm(a_init);
    opts.A_init(:,k) = a_init;
   
end

opts.X_init = zeros(m,K); % initialize X
opts.b_init = mean(y);  %b��ʼ��Ϊy�ľ�ֵ
opts.W = ones(m,K); % initialize the weight matrix

%% run the optimization algorithms
Alg_num = 4;

% Alg_type = {'ADM','iADM','homotopy-ADM','homotopy-iADM','reweighting'};
Alg_type = {'iADM'};
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
end

