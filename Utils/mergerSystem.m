function RSS = mergerSystem(price_post,x,theta_d_GMM,xi_GMM,t,J,market,OF_merge,mc_hat_comp,nume,deno,market_share_hat,market_share_hat_matrix,Qp)

    % Loop over markets and products
        for j=1:J
            utility=theta_d_GMM(1,1)+theta_d_GMM(2,1)*x((t-1)*J+j,1)+theta_d_GMM(3,1)*price_post(j,1)+xi_GMM((t-1)*J+j,1);
            nume(1,j)=exp(utility);
            deno(1,1)=deno(1,1)+nume(1,j);
        end
        for j=1:J
            market_share_hat_matrix(1,j)=nume(1,j)./deno(1,1);
            market_share_hat(j,1)=market_share_hat_matrix(1,j);
        end
    
    % ESTIMATED JACOBIAN DEMAND
    for t=1
        for j=1:J
            Qp(market==t,j)=abs(theta_d_GMM(3,1))*(market_share_hat(market==t,1).*market_share_hat_matrix(t,j));
        end
        Qp(market==t,:)=Qp(market==t,:).*(ones(J,J)-eye(J));
        Qp(market==t,:)=Qp(market==t,:)+diag(theta_d_GMM(3,1)*(market_share_hat(market==t,1).*(ones(J,1)-market_share_hat(market==t,1))));
    end
    
    % PRICE-COST MARGIN UNDER COMPETITION
    markup_hat_compet=zeros((J),1);
    for t=1
        markup_hat_compet(market==t,1)=-(Qp(market==t,:).*OF_merge)\market_share_hat(market==t,1);
    end

    RSS = price_post + markup_hat_compet(:) - mc_hat_comp(market==t,1);
end