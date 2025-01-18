clc; clear; close all;
cd('C:\Users\eliot\Documents\REPOSITORIES\EIO_project');
addpath('C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Data\');
addpath('C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Utils\');
addpath('C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Outputs\');
outputPath = 'C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Outputs\';

%%%%%%%%%%%%%%
% QUESTION 1
%%%%%%%%%%%%%%
rng(123); % Set the seed
load('Data_exam_pre.mat');
data_table=table(firm(:),product(:),market(:),price(:),quantity(:),x,quantity_0(:),v(:),...
    'VariableNames',{'firm','product','market','price','quantity','x','quantity_0','v'});
[groups,~,idx]=unique([firm,product],'rows');
Price_mean=zeros(size(groups,1),1);
Price_min=zeros(size(groups,1),1);
Price_max=zeros(size(groups,1),1);
Price_sd=zeros(size(groups,1),1);
Quantity_mean=zeros(size(groups,1),1);
Quantity_min=zeros(size(groups,1),1);
Quantity_max=zeros(size(groups,1),1);
Quantity_sd=zeros(size(groups,1),1);

for i=1:size(groups,1)
    indices=idx==i;
    Price_mean(i)=mean(price(indices));
    Price_min(i)=min(price(indices));
    Price_max(i)=max(price(indices));
    Price_sd(i)=std(price(indices));
    Quantity_mean(i)=mean(quantity(indices));
    Quantity_min(i)=min(quantity(indices));
    Quantity_max(i)=max(quantity(indices));
    Quantity_sd(i)=std(quantity(indices));
end

summary_table=table(groups(:,1),groups(:,2),Price_mean,Price_min,Price_max,Price_sd,...
    Quantity_mean,Quantity_min,Quantity_max,Quantity_sd,...
    'VariableNames',{'firm','product','Price_mean','Price_min','Price_max',...
    'Price_sd','Quantity_mean','Quantity_min','Quantity_max','Quantity_sd'});

fileID=fopen([outputPath,'summary_table.tex'],'w');
fprintf(fileID,'\\begin{tabular}{lcc|cccc|cccc}\n');
fprintf(fileID,'\\hline\n');
fprintf(fileID,'& & & \\multicolumn{4}{c|}{Price} & \\multicolumn{4}{c}{Quantity} \\\\\n');
fprintf(fileID,'Firm & Product & & Mean & Min & Max & SD & Mean & Min & Max & SD \\\\\n');
fprintf(fileID,'\\hline\n');
for i=1:height(summary_table)
    fprintf(fileID,'%d & %d & & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\\n',...
        summary_table.firm(i),summary_table.product(i),...
        summary_table.Price_mean(i),summary_table.Price_min(i),...
        summary_table.Price_max(i),summary_table.Price_sd(i),...
        summary_table.Quantity_mean(i),summary_table.Quantity_min(i),...
        summary_table.Quantity_max(i),summary_table.Quantity_sd(i));
end
fprintf(fileID,'\\hline\n');
fprintf(fileID,'\\end{tabular}\n');
fprintf(fileID,'\\end{table}\n');
fclose(fileID);
clear Price_mean Price_min Price_max Price_sd Quantity_mean Quantity_min Quantity_max Quantity_sd groups idx ans fileID i indices summary_table;

%%%%%%%%%%%%%%
% QUESTION 2
%%%%%%%%%%%%%%
load('Data_exam_pre.mat');
firm_products={1:5,6:10,11:13,14:20};
num_firms=length(firm_products);
quantity_market=zeros(height(market),1);

for t=1:T_pre
    market_indices=market==t;
    total_qt=sum(quantity(market_indices));
    quantity_market(market_indices)=total_qt;
end

product_share=quantity./quantity_market;
firm_share=zeros(T_pre,num_firms);

for t=1:T_pre
    market_indices=market==t;
    for f=1:num_firms
        firm_indices=market_indices&ismember(product,firm_products{f});
        firm_share_value=sum(product_share(firm_indices));
        firm_share(t,f)=firm_share_value;
    end
end

average_firm_share=mean(firm_share,1)*100;
HHI_pre=sum(average_firm_share.^2);
fprintf('Pre-merger HHI: %.4f\n',HHI_pre);
HHI_post=HHI_pre+2*average_firm_share(3)*average_firm_share(4);
fprintf('Post-merger HHI: %.4f\n',HHI_post);
diff=HHI_post-HHI_pre;
fprintf('Difference in HHI: %.4f\n',diff);

%%%%%%%%%%%%%%
% QUESTION 3
%%%%%%%%%%%%%%
clearvars -except outputPath
load('Data_exam_pre.mat');
total_quantity=zeros(height(market),1);

for t=1:T_pre
    market_indices=market==t;
    total_qt=sum(quantity(market_indices));
    total_quantity(market_indices)=total_qt+quantity_0(market_indices);
end

product_share=quantity./total_quantity;
product_share_0=quantity_0./total_quantity;
delta_output=log(product_share)-log(product_share_0);

X_d=[ones(size(market,1),1),x,price];
theta_MNL_OLS=(X_d'*X_d)\X_d'*delta_output;
res_MNL_OLS=delta_output-X_d*theta_MNL_OLS;
VAR_MNL_OLS=((res_MNL_OLS'*res_MNL_OLS)/(size(X_d,1)-size(X_d,2)))*((X_d'*X_d)\eye(size(X_d,2)));
SE_MNL_OLS=sqrt(diag(VAR_MNL_OLS));
CIlow_MNL_OLS=theta_MNL_OLS-SE_MNL_OLS*1.96;
CIupp_MNL_OLS=theta_MNL_OLS+SE_MNL_OLS*1.96;

RowNames={'Constant','Beta','Alpha'};
VariableNames={'Coef','SE','CI_low','CI_upp'};
results_table=table(theta_MNL_OLS,SE_MNL_OLS,CIlow_MNL_OLS,CIupp_MNL_OLS,...
    'RowNames',RowNames,'VariableNames',VariableNames);

fileID=fopen([outputPath,'regression_results_ols.tex'],'w');
fprintf(fileID,'\\begin{tabular}{lcccc}\n');
fprintf(fileID,'\\hline\n');
fprintf(fileID,'Variable & Coef & SE & CI$_{low}$ & CI$_{upp}$ \\\\\n');
fprintf(fileID,'\\hline\n');
for i=1:height(results_table)
    fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f \\\\\n',...
        RowNames{i},results_table{i,'Coef'},results_table{i,'SE'},...
        results_table{i,'CI_low'},results_table{i,'CI_upp'});
end
fprintf(fileID,'\\hline\n');
fprintf(fileID,'\\end{tabular}\n');
fclose(fileID);
disp('Regression results exported to "regression_results_ols.tex".');


%%%%%%%%%%%%%
% QUESTION 4
%%%%%%%%%%%%%
% BLP INSTRUMENT
% IN-BLP
blp_in=zeros(height(market),1);
for i=1:height(market)
    current_firm=firm(i);
    current_market=market(i);
    current_product=product(i);
    same_firm_market=(firm==current_firm)&(market==current_market);
    exclude_self=product~=current_product;
    blp_in(i)=sum(x(same_firm_market&exclude_self));
end

% OUT-BLP
blp_out=zeros(height(market),1);
for i=1:height(market)
    current_firm=firm(i);
    current_market=market(i);
    not_same_firm_market=(firm~=current_firm)&(market==current_market);
    blp_out(i)=sum(x(not_same_firm_market));
end

% GMM WITH COST SHIFTER ONLY
Zexcluded_1=[v];
Z_d=[ones(size(market,1),1) x Zexcluded_1];
W=(Z_d'*Z_d)\eye(size(Z_d,2));
theta_d_GMM=(X_d'*Z_d*W*Z_d'*X_d)\X_d'*Z_d*W*Z_d'*delta_output;
xi_GMM=delta_output-X_d*theta_d_GMM;
S=Z_d'*diag(xi_GMM.^2)*Z_d;
VAR_d_GMM=(X_d'*Z_d*W*(X_d'*Z_d)')\X_d'*Z_d*W*S*W*(X_d'*Z_d)'/(X_d'*Z_d*W*(X_d'*Z_d)');
SE_d_GMM=sqrt(diag(VAR_d_GMM));
CIlow_d_GMM=theta_d_GMM-SE_d_GMM*1.96;
CIupp_d_GMM=theta_d_GMM+SE_d_GMM*1.96;

RowNames={'Constant','Beta','Alpha'};
VariableNames={'Coef','SE','CI_low','CI_upp'};
results_table=table(theta_d_GMM,SE_d_GMM,CIlow_d_GMM,CIupp_d_GMM,...
    'RowNames',RowNames,'VariableNames',VariableNames);

fileID=fopen([outputPath,'gmm_results_cost.tex'],'w');
fprintf(fileID,'\\begin{tabular}{lcccc}\n');
fprintf(fileID,'\\hline\n');
fprintf(fileID,'Variable & Coef & SE & CI$_{low}$ & CI$_{upp}$ \\\\\n');
fprintf(fileID,'\\hline\n');
for i=1:height(results_table)
    fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f \\\\\n',...
        RowNames{i},results_table{i,'Coef'},results_table{i,'SE'},...
        results_table{i,'CI_low'},results_table{i,'CI_upp'});
end
fprintf(fileID,'\\hline\n');
fprintf(fileID,'\\end{tabular}\n');
fprintf(fileID,'\\end{table}\n');
fclose(fileID);

% GMM WITH BLP INSTRUMENTS ONLY
Zexcluded_2=[blp_in blp_out];
Z_d=[ones(size(market,1),1) x Zexcluded_2];
W=(Z_d'*Z_d)\eye(size(Z_d,2));
theta_d_GMM2=(X_d'*Z_d*W*Z_d'*X_d)\X_d'*Z_d*W*Z_d'*delta_output;
xi_GMM2=delta_output-X_d*theta_d_GMM2;
S=Z_d'*diag(xi_GMM2.^2)*Z_d;
VAR_d_GMM=(X_d'*Z_d*W*(X_d'*Z_d)')\X_d'*Z_d*W*S*W*(X_d'*Z_d)'/(X_d'*Z_d*W*(X_d'*Z_d)');
SE_d_GMM=sqrt(diag(VAR_d_GMM));
CIlow_d_GMM=theta_d_GMM2-SE_d_GMM*1.96;
CIupp_d_GMM=theta_d_GMM2+SE_d_GMM*1.96;

% WEAK INSTRUMENT TEST: F-stat 
theta_price_MNL  = (Z_d'*Z_d) \ Z_d' * price; % Regress the endogeneous variable p on instruments.
res_price_MNL    = price - Z_d * theta_price_MNL;
VARCOV_price_MNL = ( (res_price_MNL' * res_price_MNL) / (size(Z_d,1) - size(Z_d,2)) ) * ((Z_d'*Z_d) \ eye(size(Z_d,2)));
H = zeros(size(Z_d,2),size(Z_d,2));
i = 1;
for j = 1 : size(Zexcluded_2,2)
    H(i,j) = 1;
    i      = i + 1;
end
c = zeros(size(Z_d,2),1);
%[p,F_stat_MNL]=linhyptest(theta_price_MNL,VARCOV_price_MNL,c,H,size(Z_d,1)-size(Z_d,2));
%display(sprintf('\n 1st F-stat: %f %d',F_stat_MNL,p)) 
%clear theta_price_MNL res_price_MNL VARCOV_price_MNL se_price_MNL T_test_price_MNL pvalue_price_MNL  H i c p


% PERFORMING HANSEN TEST OF OVER-IDENTIFICATION
hansenTestGMM(X_d, Z_d, xi_GMM2)

RowNames={'Constant','Beta','Alpha'};
VariableNames={'Coef','SE','CI_low','CI_upp'};
results_table=table(theta_d_GMM2,SE_d_GMM,CIlow_d_GMM,CIupp_d_GMM,...
    'RowNames',RowNames,'VariableNames',VariableNames);

fileID=fopen([outputPath,'gmm_results_blp.tex'],'w');
fprintf(fileID,'\\begin{tabular}{lcccc}\n');
fprintf(fileID,'\\hline\n');
fprintf(fileID,'Variable & Coef & SE & CI$_{low}$ & CI$_{upp}$ \\\\\n');
fprintf(fileID,'\\hline\n');
for i=1:height(results_table)
    fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f \\\\\n',...
        RowNames{i},results_table{i,'Coef'},results_table{i,'SE'},...
        results_table{i,'CI_low'},results_table{i,'CI_upp'});
end
fprintf(fileID,'\\hline\n');
fprintf(fileID,'\\end{tabular}\n');
fprintf(fileID,'\\end{table}\n');
fclose(fileID);

% GMM WITH ALL INSTRUMENTS
Z_d=[ones(size(market,1),1) x Zexcluded_1 Zexcluded_2];
W=(Z_d'*Z_d)\eye(size(Z_d,2));
theta_d_GMM3=(X_d'*Z_d*W*Z_d'*X_d)\X_d'*Z_d*W*Z_d'*delta_output;
xi_GMM3=delta_output-X_d*theta_d_GMM3;
S=Z_d'*diag(xi_GMM3.^2)*Z_d;
VAR_d_GMM=(X_d'*Z_d*W*(X_d'*Z_d)')\X_d'*Z_d*W*S*W*(X_d'*Z_d)'/(X_d'*Z_d*W*(X_d'*Z_d)');
SE_d_GMM=sqrt(diag(VAR_d_GMM));
CIlow_d_GMM=theta_d_GMM3-SE_d_GMM*1.96;
CIupp_d_GMM=theta_d_GMM3+SE_d_GMM*1.96;

% PERFORMING HANSEN TEST OF OVER-IDENTIFICATION
hansenTestGMM(X_d, Z_d, xi_GMM3)



RowNames={'Constant','Beta','Alpha'};
VariableNames={'Coef','SE','CI_low','CI_upp'};
results_table=table(theta_d_GMM3,SE_d_GMM,CIlow_d_GMM,CIupp_d_GMM,...
    'RowNames',RowNames,'VariableNames',VariableNames);

fileID=fopen([outputPath,'gmm_results_cost_blp.tex'],'w');
fprintf(fileID,'\\begin{tabular}{lcccc}\n');
fprintf(fileID,'\\hline\n');
fprintf(fileID,'Variable & Coef & SE & CI$_{low}$ & CI$_{upp}$ \\\\\n');
fprintf(fileID,'\\hline\n');
for i=1:height(results_table)
    fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f \\\\\n',...
        RowNames{i},results_table{i,'Coef'},results_table{i,'SE'},...
        results_table{i,'CI_low'},results_table{i,'CI_upp'});
end
fprintf(fileID,'\\hline\n');
fprintf(fileID,'\\end{tabular}\n');
fprintf(fileID,'\\end{table}\n');
fclose(fileID);


%%%%%%%%%%%%%%%%%%%%%
% SUPPLY - QUESTION 5
%%%%%%%%%%%%%%%%%%%%%
% MARKET SHARE ESTIMATED FROM LOGIT
nume=zeros(T_pre,J);
deno=zeros(T_pre,1);
deno=exp(deno);
market_share_hat_matrix=zeros(T_pre,J);
market_share_hat=zeros(T_pre*J,1);

% Loop over markets and products
for t=1:T_pre
    for j=1:J
        idx=(t-1)*J+j; % Linear index for market-share-related variables
        utility=theta_d_GMM(1,1)+theta_d_GMM(2,1)*x((t-1)*J+j,1)+theta_d_GMM(3,1)*price((t-1)*J+j,1)+xi_GMM((t-1)*J+j,1);
        nume(t,j)=exp(utility);
        deno(t,1)=deno(t,1)+nume(t,j);
    end
    for j=1:J
        market_share_hat_matrix(t,j)=nume(t,j)./deno(t,1);
        market_share_hat((t-1)*J+j,1)=market_share_hat_matrix(t,j);
    end
end

% OWNERSHIP MATRIX WHEN COMPETITION
OF=zeros(J,J);
firm_matrix=customDummyVar(firm);
for j=1:J
    OF(j,:)=firm_matrix(market==t,firm(market==t&product==j,1))';
end

% OWNERSHIP MATRIX WHEN FIRMS 3 and 4 COLLUDE
OFcoll=OF;
OFcoll(11:end,11:end)=1;

% ESTIMATED JACOBIAN DEMAND
Qp=zeros(J*T_pre,J);
for t=1:T_pre
    for j=1:J
        Qp(market==t,j)=abs(theta_d_GMM(3,1))*(market_share_hat(market==t,1).*market_share_hat_matrix(t,j));
    end
    Qp(market==t,:)=Qp(market==t,:).*(ones(J,J)-eye(J));
    Qp(market==t,:)=Qp(market==t,:)+diag(theta_d_GMM(3,1)*(market_share_hat(market==t,1).*(ones(J,1)-market_share_hat(market==t,1))));
end

% PRICE-COST MARGIN UNDER COMPETITION
markup_hat_compet=zeros((J*T_pre),1);
for t=1:T_pre
    markup_hat_compet(market==t,1)=-(Qp(market==t,:).*OF)\market_share_hat(market==t,1);
end

% PRICE-COST MARGIN UNDER COLLUSION
markup_hat_col=zeros((J*T_pre),1);
for t=1:T_pre
    markup_hat_col(market==t,1)=-(Qp(market==t,:))\market_share_hat(market==t,1);
end

% HISTOGRAM OF PRICE COST MARGINS
figure;
subplot(1,2,1);
histogram(markup_hat_compet,'BinWidth',0.01,'FaceAlpha',0.7,'EdgeColor','none');
xlabel('Markup (Competition)','FontSize',8);
ylabel('Count','FontSize',8);
grid on;

subplot(1,2,2);
histogram(markup_hat_col,'BinWidth',0.05,'FaceAlpha',0.7,'EdgeColor','none');
xlabel('Markup (Full collusion)','FontSize',8);
ylabel('Count','FontSize',8);
saveas(gcf,[outputPath,'markup_distribution_collusion.png']);

% MARGINAL COST UNDER COMPETITION
mc_hat_comp=zeros((J*T_pre),1);
for t=1:T_pre
mc_hat_comp(market==t,1) = price(market==t,1) + (Qp(market==t,:).*OF) \ market_share_hat(market==t,1);
end

% MARGINAL COST UNDER COLLUSION
mc_hat_col=zeros((J*T_pre),1);
for t=1:T_pre
mc_hat_col(market==t,1) = price(market==t,1) + (Qp(market==t,:)) \ market_share_hat(market==t,1);
end

% HISTOGRAMS OF MARGINAL COST
figure;
subplot(1,3,1);
histogram(mc_hat_comp,'BinWidth',0.01,'FaceAlpha',0.7,'EdgeColor','none');
xlabel('Marginal costs (Competition)','FontSize',8);
ylabel('Count','FontSize',8);
grid on;

subplot(1,2,2);
histogram(mc_hat_col,'BinWidth',0.05,'FaceAlpha',0.7,'EdgeColor','none');
xlabel('Marginal costs (Full collusion)','FontSize',8);
ylabel('Count','FontSize',8);
saveas(gcf,[outputPath,'mc_distribution_collusion.png']);

%%%%%%%%%%%%
% QUESTION 7
%%%%%%%%%%%%

% OWNERSHIP MATRIX AFTER FIRM 3 AND FIRM 4 MERGE
OF_merge=zeros(J,J); % Initialize the ownership matrix
firm_matrix_collusion=customDummyVar(firm); % Create dummy variable matrix for firm ownership

% Loop through products to fill the ownership matrix
for j=1:J
    % Get the firm for the current product in market t
    current_firm=firm(market==t&product==j,1);

    if current_firm==3 || current_firm==4
        colluding_firms=[3,4];
        OF_merge(j,:)=sum(firm_matrix_collusion(market==t,colluding_firms),2)';
    else
        % For other firms, retain the actual ownership
        OF_merge(j,:)=firm_matrix_collusion(market==t,current_firm)';
    end
end

% PRICE-COST MARGIN UNDER COMPETITION
markup_hat_merge_approx=zeros((J*T_pre),1);
for t=1:T_pre
    markup_hat_merge_approx(market==t,1)=-(Qp(market==t,:).*OF_merge)\market_share_hat(market==t,1);
end

% NEW PRICE
price_merge_approx = zeros(J*T_pre,1);
for t=1:T_pre
    price_merge_approx(market==t,1)=mc_hat_comp(market==t,1)+markup_hat_merge_approx(market==t);
end

data_table_sim=table(market,price,price_merge_approx,firm,product);
data_table_sim.price_change=(data_table_sim.price_merge_approx-data_table_sim.price)./data_table_sim.price;

average_price_change = zeros(4,1);
for i=1:4
    firm_id=i; % Current firm ID
    % Get the rows corresponding to the current firm
    firm_indices=(data_table_sim.firm==firm_id);
    % Compute the average price change for the current firm
    average_price_change(i)=mean(data_table_sim.price_change(firm_indices));
end

%%%%%%%%%%%%%%
% QUESTION 8
%%%%%%%%%%%%%%

% SOLVING USING fminunc AND quasi-newton algorithm
close all
price_merge = zeros(J*T_pre,1);
for t = 1:T_pre
    price_merge(J*(t-1)+1:J*t,1) = solveP(price(market==t),x,theta_d_GMM,xi_GMM,t,J,market,OF_merge,mc_hat_comp);
end
histogram(price_merge) 
% pb 1: some prices negatives
% pb 2: difficulty computing inverse of (Qp(:,:).*OF_merge)


% SOLVING USING fsolve
% setting up objects
nume=zeros(1,J);
deno=zeros(1,1);
deno=exp(deno);
market_share_hat_matrix=zeros(1,J);
market_share_hat=zeros(1*J,1);
Qp=zeros(J*1,J);

price_merge2 = zeros(J*T_pre,1);
for t = 1:T_pre
    % initialise prices at pre-merger
    price_0 = price(market==t); 
    options = optimoptions('fsolve', ...
        'TolFun', 1e-14, ...   % Function tolerance
        'TolX', 1e-14, ...     % Step tolerance
        'Display', 'final-detailed');   % Display iteration information
    price_merge2(J*(t-1)+1:J*t,1)  = fsolve(@(price_post) mergerSystem(price_post,x,theta_d_GMM3,xi_GMM3,t,J,market,OF_merge,mc_hat_comp,nume,deno,market_share_hat,market_share_hat_matrix,Qp), price_0, options);
end
histogram(price_merge2)



