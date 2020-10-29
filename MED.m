%% 说明
% MED 方案对比合集

clc 
clear all
close all
addpath(genpath(pwd));
%% 构建卷积核函数
t = 1:0.01:10;
% 波形参数赋值
alpha = 2;
tau = 5;
f = 2;
phi = 1;
beta = 1;
% 子波形(高斯模型)
echo = beta*exp(-alpha*(t-tau).^2).*cos(2*pi*f*(t-tau)+phi);
% 核函数
h_real = echo(250:550);
h_real = h_real/norm(h_real)+0.01;
% plot(h)

%% 构建稀疏激励
% 稀疏度
k = 3;
% 信号长度
p = 1000;
x_real = zeros(1, p);
% index = randperm(p);
% x(index(1:k)) = -1 + 2*rand(1, k);
index = [100, 400, 700];
x_real(index) = -1 + 2*rand(1, k);
x_real = abs(x_real);
% 构造循环矩阵
C = zeros(p);
for count=1:p
    C(:, count) = circshift([h_real, zeros(1, p-length(h_real))],[0,count-1]).';
end

%% 构造输入信号
Y = C * x_real.';
%% 1.MED 方案  
% 初始的MED方案解不出结果
% [y_final f_final kurt] = med2d(Y',30,100,[],'full',1);
%% 2.OMED方案 
% 可用，可以解出相对来说比较满意的结果
% [x_solved, f, d_norm] = omeda(Y,20,1);

%% 3.MOMEDA 方案

% 有很多尖峰，效果并不理想


%     window = ones(1,1);
%     
%     % 1000-sample FIR filters will be designed
%     L = 100;
%     
%     % Plot a spectrum from a period of 10 to 300 (actual fault is at 50)
%     range = [10:0.1:50];
%     
%     % Plot the spectrum
%     [T MKurt f y T_best MKurt_best f_best y_best] = momeda_spectrum(Y,L,window,range,1);

%% 4.momeda 方案 
% 有尖峰，效果不是很理想

%     % No window. A 5-sample recangular window would be ones(5,1).
%     window = ones(1,1);
%     
%     % 1000-sample FIR filters will be designed
%     L = 100;
%     
%     % Recover the fault signal of period 50
%     [MKurt f y] = momeda(Y,L,window,50,1);

%% 5.mckd方案  不太行
%          [y_final f_final ck_iter] = mckd(Y,400,30,100,7,1); % M = 7
%                                                              % T = 100






%构造MED输入，假定每行信号相同
% x_input = [Y';Y';Y'];   % 3*1000
% 
% %% 初始化f  
% f = zeros(1,100);    % 1*100行向量
% f(50)=1;
% L_f = length(f);
% L_Y = length(Y);
% [x_rows,x_clos]= size(x_input);
% y_output = zeros(3,L_Y-L_f-1);
% %% 计算V
% V = zeros(1,x_rows);
% for i = 1:x_rows
%     V_upperj = 0;
%     V_lowerj = 0;
%     for j = L_f+1:L_Y
%        %求和计算y(i,j)
%        y_output(i,j) =0;
%        for k=1:L_f
%            y_output(i,j)= y_output(i,j)+f(1,k)*x_input(i,j-k);
%        end
%        %k循环结束得到y(i,j)
%        %对于每一个j
%        V_upperj = V_upperj + y_output(i,j)^4;
%        V_lowerj = V_lowerj + y_output(i,j)^2;
%     end
%     %j循环结束，得到每一个y(i,j),计算V_j
%     V(1,i) =  V_upperj / (V_lowerj)^2;
% end
% % V_sum为方差
% V_sum = sum(V);
% 
% 
% %% 求解滤波器系数



