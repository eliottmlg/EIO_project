
function hansenTestGMM(y, x, z, beta_gmm)
    % Inputs:
    % y: Dependent variable (n x 1)
    % x: Endogenous variables (n x k)
    % z: Instrumental variables (n x m)

    % Ensure inputs are column vectors/matrices
    y = y(:); 
    [n, k] = size(x); % n: number of observations, k: number of regressors
    [n2, m] = size(z); % m: number of instruments

    if n ~= n2
        error('Dimensions of y, x, and z must match.');
    end

    % Step 5: Compute GMM residuals and moment conditions
    residuals_gmm = y - x * beta_gmm;
    g_gmm = z' * residuals_gmm / n;

    % Step 6: Optimal weight matrix (second step GMM)
    S = (z' * diag(residuals_gmm.^2) * z) / n;
    W_opt = inv(S);

    % Step 7: Re-estimate GMM with optimal weight matrix
    beta_gmm_opt = (x' * z * W_opt * z' * x) \ (x' * z * W_opt * z' * y);

    % Step 8: Compute the Hansen J-statistic
    residuals_final = y - x * beta_gmm_opt;
    g_final = z' * residuals_final / n;
    J_stat = n * g_final' * W_opt * g_final;

    % Step 9: Degrees of freedom and p-value
    df = m - k; % Degrees of freedom (number of over-identifying restrictions)
    p_value = 1 - customChi2Cdf(J_stat, df);

    % Display results
    fprintf('Hansen J-statistic: %.4f\n', J_stat);
    fprintf('Degrees of freedom: %d\n', df);
    fprintf('p-value: %.4f\n', p_value);

    % Interpretation
    if p_value > 0.05
        disp('Fail to reject the null hypothesis: Instruments are valid.');
    else
        disp('Reject the null hypothesis: Some instruments may be invalid.');
    end
end



