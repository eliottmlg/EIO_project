function [F, pValue] = customLinhyptest(X, y, R, q)
    % Inputs:
    % X: Design matrix (n x k)
    % y: Dependent variable (n x 1)
    % R: Restriction matrix (m x k)
    % q: Hypothesized values (m x 1)
    %
    % Outputs:
    % F: F-statistic
    % pValue: p-value of the test
    
    % Dimensions
    [n, k] = size(X);
    m = size(R, 1);

    % OLS estimate of beta
    beta_hat = (X' * X) \ (X' * y);

    % Residuals and RSS
    residuals = y - X * beta_hat;
    RSS = residuals' * residuals;

    % Estimate of sigma^2
    sigma_hat2 = RSS / (n - k);

    % Test statistic
    R_beta_q = R * beta_hat - q; % R * beta_hat - q
    inv_RVR = inv(R * (X' * X) \ R');
    F = (R_beta_q' * inv_RVR * R_beta_q / m) / sigma_hat2;

    % p-value
    pValue = 1 - fcdf(F, m, n - k);

    % Display results
    fprintf('F-statistic: %.4f\n', F);
    fprintf('p-value: %.4f\n', pValue);

    if pValue > 0.05
        disp('Fail to reject the null hypothesis.');
    else
        disp('Reject the null hypothesis.');
    end
end
