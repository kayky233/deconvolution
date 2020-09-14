% update X via backtracking linesearch
% backtracking 有有些看不懂了
function [X1, t] = backtracking( y, A, X, fx, grad_fx, lambda, t, opts)

m = length(y);

Q = @(Z,tau) fx + norm(lambda .* Z,1) + innerprod(grad_fx, Z-X) + 0.5/tau*norm(Z-X,'fro')^2;% 'fro' norm calculates the 2-norm of the column vector,

t = 8*t;%步数调整为8倍

X1 = soft_thres( X - t * grad_fx, lambda * t ); %proximal mapping

while ( Psi_val(y, A, X1, lambda) > Q(X1,t) )%意义不明，不太懂
    t = 0.5*t;
    X1 = soft_thres( X - t * grad_fx, lambda * t );% X更新方式，不太懂

    if(opts.isupperbound)
        X1 = min(X1,opts.upperbound);
    end
end

end



function f = innerprod(U,V)
f = sum(sum(U.*V));
end

function f = Psi_val( y, A, Z, lambda)
m = length(y);
[~,K] = size(A);
y_hat = zeros(size(y));

for k = 1:K
    y_hat = y_hat + cconv( A(:,k), Z(:,k), m);
end

f = 0.5 * norm(y - y_hat)^2 +  norm(lambda .* Z,1);

end


