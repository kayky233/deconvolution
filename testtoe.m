clc
clear all
close all
addpath(genpath(pwd));
for Iter = 1: 1
t = 1:0.01:10;
% 波形参数赋值
alpha = 2;
tau = 1.5;
f = 3;
phi = 1;
beta = 1;
% 子波形(高斯模型)
echo = beta*exp(-alpha*(t-tau).^2).*cos(2*pi*f*(t-tau)+phi);
% 核函数
h = echo(1:200);
% plot(echo)

% 稀疏度
k = 3;
p = 1e3;
x = zeros(1, p);
index = randperm(p);
x(index(1:k)) = -1 + 2*rand(1, k);
% x([200, 500, 700]) = 1;

y = conv(h, x);
y_noise = awgn(y, 15, 'measured');
% plot(h)

%[A, X] = deconvolutionmodel(y.', 200, length(y));
[A, X] = advanceddecomodel(y.', 200, length(y));
% figure(1)
% plot(h)
% hold on
% plot(A*5)
% hold off
h_toep = 5*A.';
r = [h_toep zeros(1,length(x)-1)];
c = [h_toep(1) zeros(1,length(x)-1)];
hConv = toeplitz(r,c);

[x_slove,x_debias,objective,times,debias_start,mses,max_svd] = ...
         TwIST(y.',hConv,0.5, 'lambda', 1e-4, 'StopCriterion', 3, 'ToleranceA', 0.0001);
     
     
figure()
subplot(2,1,1)
plot(h)
hold on
plot(A*5)
hold off

subplot(2,1,2)
plot(y)
hold on
plot(x_slove,'r')
plot(x,'b')
hold off

% i = i + 1 ;
% end
Iter = Iter +1;
end
