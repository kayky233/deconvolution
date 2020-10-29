function [y f d_norm] = omeda(x,filterSize,plotMode)
    % MINIMUM ENTROPY DECONVOLUTION D-NORM EXACT SOLUTION APPROACH
    %       code by Geoff McDonald (glmcdona@gmail.com), 2015
    %
    % omeda(x,filterSize,overlapMode,plotMode)
    %  A noniterative optimal MED (OMED) computation algorithm. This method
    %  solves a similar deconvolution problem to MED, and is able to directly solve for
	%  the optimal filter solution as proposed by Carlos Cabrelli. This can be used is rotating
	%  machine fault detection from vibration signals to detect gear and bearing faults, and OMED
	%  is applied to extract the impulse-like features in the vibration.
    %
    %  This implementation uses the convolution adjustment proposed by myself in the second paper
	%  reference, which is important to prevent this method from reaching the trivial solution of
	%  deconvolving the convolution discontinuity.
    %
    %  Note to readers, you may want to refer to some of my other MED-based submissions:
    %      MED:
    %             The iterative non-optimal solution is often better for vibration fault detections. It
    %             is often better than this optimal solution, since the MED problem aims to deconvolve
    %             only a single-impulse as the result. As a result, OMED is able to better-extract the
    %             solution of deconvolving only the single-impulse, whereas MED more commonly
    %             reaches the solution of deconvolving the desired impulse train.
    %      MOMEDA:
    %             This is the optimal solution to the periodic impulses deconvolution problem and is recommended
    %             for rotating machine faults instead of MED or OMED. Since it is non-iterative, it is able to
    %             quickly generate spectrum's to diagnose machine health.
    %
    % Algorithm Reference:
    %   Original derivation:
    %    C A. Cabrelli, Minimum entropy deconvolution and simplicity: A
    %    non-iterative algorithm, Geophysics, Vol. 50. No. 3. March 1984.
    % 
    %   Convolution adjustment:
    %    G.L. McDonald, Qing Zhao, Multipoint Optimal Minimum Entropy Deconvolution and Convolution
    %    Fix: Application to Vibration Fault Detection, unpublished
    %
    % Inputs:
    %    x: 
    %       Signal to perform Minimum Entropy Deconvolution on.
    % 
    %    filterSize:
    %       This is the length of the finite inpulse filter filter to 
    %       design. Using a value of around 30 is appropriate depending on
    %       the data. Investigate the performance difference using
    %       different values.
    % 
    %    plotMode:
    %       If this value is > 0, plots will be generated of the iterative
    %       performance and of the resulting signal.
    %
    % Outputs:
    %    y_final:
    %       The input signal(s) x, filtered by the resulting MED filter.
    %       This is obtained simply as: y_final = filter(f_final,1,x);
    %
    %    f_final:
    %       The final 1d MED filter in finite impulse response format.
    % 
    %    d_norm:
    %       Final D-Norm of the filtered signal.
    %
    % Example:
    % % Simple vibration fault model
    % close all
    % n = 0:1999;
    % h = [-0.05 0.1 -0.4 -0.8 1 -0.8 -0.4 0.1 -0.05];
    % faultn = 0.05*(mod(n,50)==0);
    % fault = filter(h,1,faultn);
    % noise = wgn(1,length(n),-40);
    % x = sin(2*pi*n/30) + 0.2*sin(2*pi*n/60) + 0.1*sin(2*pi*n/15) + fault;
    % xn = x + noise;
    % 
    % % 20-sample FIR filters will be designed
    % L = 20;
    % 
    % % Recover the fault signal
    % [y, f, d_norm] = omeda(xn,L,1);
    % 
    
    % Assign default values for inputs
    if( isempty(filterSize) )
        filterSize = 50;
    end
    if( isempty(plotMode) )
        plotMode = 0;
    end
    
    % Validate the inputs
    overlapMode = 'valid'; % This setting forces it to use the convolution adjusted solution and is highly recommended.
                           % I didn't provide this as an input argument since using the original 'full' mode would be
                           % erroneous in most cases.

    if( strcmp(overlapMode,'full') == 1 )
        overlap_full = 1; % not recommended
    elseif( strcmp(overlapMode,'valid') == 1 )
        overlap_full = 0;
    else
        error('OMEDA:NoOverlapMode', 'overlapMode argument must be "valid" or "full".')
    end
    
    if( sum( size(x) > 1 ) > 1 )
        error('OMEDA:InvalidInput', 'Input signal x must be 1d.')
    elseif(  sum(size(plotMode) > 1) ~= 0 )
        error('OMEDA:InvalidInput', 'Input argument plotMode must be a scalar.')
    elseif( sum(size(filterSize) > 1) ~= 0 || filterSize <= 0 || mod(filterSize, 1) ~= 0 )
        error('OMEDA:InvalidInput', 'Input argument filterSize must be a positive integer scalar.')
    end
    
    L = filterSize;
    x = x';
    
    % If the data is 1d, lets make it a column vector
    if( sum(size(x)>1) == 1 )
        x = x(:); % A column vector
    end    
    
    % Calculate X0
    N = length(x);
    X0 = zeros(L,N+L-1); % y = f*x where x is padded
    for( l =1:L )
        if( l == 1 )
            X0(l,1:N) = x(1:N);
        else
            X0(l,2:end) = X0(l-1, 1:end-1);
        end
    end    
    if( ~overlap_full )     % "valid" region only. This fixes the convolution definition.
        X0 = X0(:,L:N-1);   % y = f*x where only valid x is used
                            % y = X0'x to get valid output signal
    end

    % Calculate the inverse component
    autocorr_inv = pinv(X0*X0');
    
    % Now we need to calculate all the possible filters solutions
    d_norms = zeros(size(X0,2),1); % Holds the D-Norms
    k_norms = zeros(size(X0,2),1); % Holds the Kurtosis Norms
    
    d_norm_best = -1;
    f_best = zeros(filterSize,1);
    j_best = 0;
    
    % Calculate the filter solution matrix
    F = autocorr_inv * X0;
    Y = X0' * F;

    % Find the best-result filter as the optimal answer
    for j = 1:size(X0,2)
        norm = dnorm(Y(:,j),j);

        if( norm > d_norm_best )
            % Remember this best result
            j_best = j;
            d_norm_best = norm;
            kurt_best = kurtosis(Y(:,j));
            f_best = F(:,j);
            y_best = Y(:,j);
        end

        % Store this position result
        k_norms(j) = kurtosis(Y(:,j));
        d_norms(j) = norm;
    end
    
    % Select the best result as the optimal solution answer
    f = f_best;
    y = y_best;
    d_norm = d_norm_best;

    % Plot the results
    if( plotMode > 0 )
        figure;
        subplot(5,1,1)
        plot(d_norms')
        hold on
        stem(j_best,d_norms(j_best),'black');
        hold off
        title('D-Norm')
        
        subplot(5,1,2)
        plot(k_norms')
        hold on
        stem(j_best,k_norms(j_best),'black');
        hold off
        title('Kurtosis')

        subplot(5,1,3)
        plot(y)
        title('Output')

        subplot(5,1,4)
        plot(x)
        title('Input')

        subplot(5,1,5)
        stem(f)
        title('Filter')
    end
end


function [result] = dnorm(x,k)
    result = abs(x(k))/(sum(sum(x.^2))^(1/2));
end

function [result] = kurtosis(x)
    % This function simply calculates the summed kurtosis of the input
    % signal, x.
    result = sum(x.^4)/(sum(x.^2)^2);
end
