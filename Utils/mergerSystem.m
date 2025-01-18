function RSS = mergerSystem(price_post,x,theta_d_GMM3,xi_GMM3,t,J,market,OF_merge,mc_hat_comp,nume,deno,market_share_hat,market_share_hat_matrix,Qp)

    for j=1:J
        utility=theta_d_GMM3(1,1)+theta_d_GMM3(2,1)*x((t-1)*J+j,1)+theta_d_GMM3(3,1)*price_post(j,1)+xi_GMM3((t-1)*J+j,1);
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

    % criterion to minimise
    RSS = price_post + markup_hat_merge(:) - mc_hat_comp(market==t,1);

end

