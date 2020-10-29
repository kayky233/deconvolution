function [T MKurt f y T_best MKurt_best f_best y_best] = momeda_spectrum(x,filterSize,window,range,plotMode)
    % MULTIPOINT OPTIMAL MINUMUM ENTROPY DECONVOLUTION ADJUSTED
    %       code by Geoff McDonald (glmcdona@gmail.com), 2015
    %       Used in my research with reference to an unpublished paper.
    %
    % momeda_spectrum(x,filterSize,window,range,plotMode)
    %  Multipoint Optimal Minimum Entropy Deconvolution (MOMEDA) computation algorithm. This proposed
    %  method solves the optmial solution for deconvolving a periodic train of impulses from a signal.
    %  It is best-suited in application to rotating machine faults from vibration signals, to deconvolve
    %  the impulse-like vibration associated with many gear and bear faults. This function generates
    %  a spectrum of how-well impulse-trains can be deconvolved at each period of separation between
    %  the impulses. Generally, this spectrum will product peaks at periods corresponding to critial
    %  frequencies, and the resulting magnitudes may be tracked to monitor machine component health.
    %
    %  This method is derived in the Algorithm Reference section.
    %
    % Inputs:
    %    x: 
    %       Signal to generate the MOMEDA spectrum on. Generally this should be around the range
    %       of 1000 to 10,000 samples covering at least 5 rotations of the elements in the machine.
    % 
    %    filterSize:
    %       This is the length of the finite inpulse filter filter to 
    %       design. This must be larger than max(range). Generally a number
    %       on the order of 500 or 1000 is good, but may depend on the
    %       dataset length.
    % 
    %    window:
    %       This is the window that be convolved with the impulse train target. Generally, a
    %       rectangular window works well, eg [1 1 1 1 1]. Has to be shorter in length
    %       than min(range).
    % 
    %    range:
    %       This is the periods to test as the spectrum x-axis. It should be a decimal range, like:
    %           range = 5:0.1:300;
    % 
    %    plotMode:
    %       If this value is > 0, plots will be generated of the iterative
    %       performance and of the resulting signal.
    %
    % Outputs:
    %    T:
    %       The x-axis of the resulting spectrum, representing the period in samples.
    %
    %    MKurt:
    %       The y-axis of the spectrum, representing the Multipoint Kurtosis of the Deconvolution
    %       result at the corresponding period in T.
    %
    %    f:
    %       Optimal filters designed for each period in T.
    %
    %    y:
    %       Outputs for each filter designe at each period in T.
    % 
    %    T_best:
    %       Period T corresponding to the highest MKurt.
    %
    %    MKurt_best:
    %       max(MKurt)
    %
    %    f_best
    %       Filter corresponding to maximum MKurt in the range provided.
    %
    %    y_best
    %       Most-faulty output signal, max(MKurt).
    %
    % Example:
    %
    % % Simple vibration fault model
    % close all
    % n = 0:9999;
    % h = [-0.05 0.1 -0.4 -0.8 1 -0.8 -0.4 0.1 -0.05];
    % faultn = 0.05*(mod(n,50)==0);
    % fault = filter(h,1,faultn);
    % noise = wgn(1,length(n),-25);
    % x = sin(2*pi*n/30) + 0.2*sin(2*pi*n/60) + 0.1*sin(2*pi*n/15) + fault;
    % xn = x + noise;
    % 
    % % No window. A 5-sample recangular window would be ones(5,1)
    % window = ones(1,1);
    % 
    % % 1000-sample FIR filters will be designed
    % L = 1000;
    % 
    % % Plot a spectrum from a period of 10 to 300 (actual fault is at 50)
    % range = [10:0.1:300];
    % 
    % % Plot the spectrum
    % [T MKurt f y T_best MKurt_best f_best y_best] = momeda_spectrum(xn,L,window,range,1);
    % 
    % % Now lets extract the fault signal, assuming we know it has a period between 45 and 55
    % window = ones(1,1); % this is no window
    % range = [45:0.1:55];
    % [T MKurt f y T_best MKurt_best f_best y_best] = momeda_spectrum(xn,L,window,range,0);
    % 
    % % Plot the resulting fault signal
    % figure;
    % plot( y_best(1:1000) );
    % title(strcat(['Extracted fault signal (period=', num2str(T_best), ')']))
    %     
    
    % Assign default values for inputs
    if( isempty(filterSize) )
        filterSize = 300;
    end
    if( isempty(plotMode) )
        plotMode = 0;
    end
    if( isempty(window) )
        window = ones(1,1);
    end
    if( isempty(range) )
        range = [5:0.05:300];
    end
    
    if( sum( size(x) > 1 ) > 1 )
        error('MOMEDA:InvalidInput', 'Input signal x must be 1d.')
    elseif(  sum(size(plotMode) > 1) ~= 0 )
        error('MOMEDA:InvalidInput', 'Input argument plotMode must be a scalar.')
    elseif( sum(size(filterSize) > 1) ~= 0 || filterSize <= 0 || mod(filterSize, 1) ~= 0 )
        error('MOMEDA:InvalidInput', 'Input argument filterSize must be a positive integer scalar.')
    elseif( sum(size(window) > 1) > 1 )
        error('MOMEDA:InvalidInput', 'Input argument window must be 1d.')
    elseif( min(range) <= length(window) )
        error('MOMEDA:InvalidInput', 'Range starting point must be larger than the length of the window.')
    elseif( filterSize >= length(x) )
        error('MOMEDA:InvalidInput', 'Input argument filterSize must be smaller than the length of input signal x.')
    end
    
    L = filterSize;
    x = x(:); % A column vector
    
    %%% Calculte X0 matrix
    N = length(x);
    X0 = zeros(L,N);
    
    for( l =1:L )
        if( l == 1 )
            X0(l,1:N) = x(1:N);
        else
            X0(l,2:end) = X0(l-1, 1:end-1);
        end
    end
    
                        % "valid" region only
    X0 = X0(:,L:N-1);   % y = f*x where only valid x is used
                        % y = Xm0'*x to get valid output signal
    
    autocorr = X0*X0';
    autocorr_inv = pinv(autocorr);
    
    % Built the array of targets impulse train vectors separated the by periods
    T = zeros(length(range),1);
    i = 1;
    t = zeros(N-L,length(range));
    for period = range
        points{i} = 1:period:(size(X0,2)-1);
        points{i} = round(points{i});
        t(points{i},i) = 1;
        T(i) = period;
        i = i + 1;
    end
    
    % Apply the windowing function to the target vectors
    t = filter(window, 1, t);
    
    % Calculate the spectrum of optimal filters
    f = autocorr_inv * X0 * t;

    % Calculate the spectrum of outputs
    y = X0'*f;
    
    % Calculate the spectrum of PKurt values for each output
    MKurt = mkurt(y,t);
    
    % Find the best match
    [MKurt_best index_max] = max(MKurt);
    T_best = T(index_max);
    f_best = f(:,index_max);
    y_best = y(:,index_max);
    
    % Plot the resulting spectrum
    if( plotMode > 0 )
        figure;
        plot(T,MKurt);
        ylabel('Multipoint Kurtosis')
        xlabel('Period (samples)');
        axis('tight')
        
        figure;
        subplot(3,1,1)
        plot(x)
        title('Input signal');
        xlabel('Sample number');
        
        subplot(3,1,2)
        plot(y_best)
        title(strcat(['Best output signal (period=', num2str(T_best), ')']));
        xlabel('Sample number');
        
        subplot(3,1,3)
        stem(f_best)
        title(strcat(['Best filter (period=', num2str(T_best), ')']));
        xlabel('Sample number');
    end
end

function [result] = mkurt(x,target)
    % This function simply calculates the summed kurtosis of the input
    % signal, x, according to the target vector positions.
    result = zeros(size(x,2),1);
    for i = 1:size(x,2)
        result(i) = ( (target(:,i).^4)'*(x(:,i).^4) )/(sum(x(:,i).^2)^2) * sum(abs(target(:,i)));
    end
end
