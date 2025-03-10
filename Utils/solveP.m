function [price_post] = solveP(price_pre,x,theta_d_GMM3,xi_GMM3,t,J,market,OF_merge,mc_hat_comp)

    price_0 = price_pre;
    
    options    = optimoptions(@fminunc,'Algorithm','quasi-newton','Display','none','MaxFunEvals',Inf,'MaxIter',Inf,'TolX',1e-14,'TolFun',1e-14);
    price_post = fminunc(@OLS_optim,price_0,options);
    
    function RSS = OLS_optim(price_iter)

        % market share %
        nume=zeros(1,J);
        deno=zeros(1,1);
        deno=exp(deno);
        market_share_hat_matrix=zeros(1,J);
        market_share_hat=zeros(1*J,1);
        Qp=zeros(J*1,J);

        for j=1:J
            utility=theta_d_GMM3(1,1)+theta_d_GMM3(2,1)*x((t-1)*J+j,1)+theta_d_GMM3(3,1)*price_iter(j,1)+xi_GMM3((t-1)*J+j,1);
            nume(j)=exp(utility);
            deno(1)=deno(1)+nume(j);
        end
        for j=1:J
            market_share_hat_matrix(j)=nume(j)./deno(1);
            market_share_hat(j,1)=market_share_hat_matrix(j);
        end

        % ESTIMATED JACOBIAN DEMAND
        for j=1:J
            Qp(:,j)=abs(theta_d_GMM3(3,1))*(market_share_hat(1).*market_share_hat_matrix(j));
        end
        Qp(:,:)=Qp(:,:).*(ones(J,J)-eye(J));
        Qp(:,:)=Qp(:,:)+diag(theta_d_GMM3(3,1)*(market_share_hat(:,1).*(ones(J,1)-market_share_hat(:,1))));

        % markup
        markup_hat_merge=zeros((J*1),1);
        markup_hat_merge(:,1)=-(Qp(:,:).*OF_merge)\market_share_hat(:,1);

        % criterion to minimise: sup norm of RSS
        RSS = price_iter + markup_hat_merge(:) - mc_hat_comp(market==t,1);
        RSS = max(sum(abs(RSS), 2));
    
    end
    
end