
function hansenTestGMM(X_d, Z_d, xi_GMM)
    W = (Z_d' * diag(xi_GMM.^2) * Z_d) \ eye(size(Z_d,2));
    if size(Z_d,2)-(size(X_d,2)+1) > 0
        OIR       = xi_GMM' * Z_d * W * (xi_GMM' * Z_d)';
        pvalueOIR = 1 - customChi2Cdf(OIR,size(Z_d,2)-(size(X_d,2)+1));
        table(OIR,pvalueOIR,'RowNames',{'Hansen_test'},'VariableNames',{'Stat','pvalue'})
    end
end



