function p = customChi2Cdf(x, k)
    % Replicates the chi2cdf function
    % Inputs:
    %   x: Value at which to evaluate the chi-square CDF (can be scalar or vector)
    %   k: Degrees of freedom
    % Output:
    %   p: Cumulative probability up to x

    % Validate inputs
    if k <= 0
        error('Degrees of freedom (k) must be positive.');
    end
    if any(x < 0)
        error('Input x must be non-negative.');
    end

    % Compute the CDF using gammainc
    % gammainc calculates the regularized incomplete gamma function
    p = gammainc(x / 2, k / 2);
end
