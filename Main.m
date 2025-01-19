clc; clear; close all;
cd('C:\Users\eliot\Documents\REPOSITORIES\EIO_project');
addpath('C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Data\');
addpath('C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Utils\');
addpath('C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Outputs\');
outputPath = 'C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Outputs\';


%%%%%%%%%%%%%%
% QUESTION 1
%%%%%%%%%%%%%%
rng(123); 
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

% Open a file to write the LaTeX table
fileID=fopen([outputPath,'summary_table.tex'],'w');

% Write the LaTeX table header with multi-row column grouping
fprintf(fileID, '\\begin{tabular}{lcc|cccc|cccc}\n');
fprintf(fileID, '\\hline\n');
fprintf(fileID, '& & & \\multicolumn{4}{c|}{Price} & \\multicolumn{4}{c}{Quantity} \\\\\n');
fprintf(fileID, 'Firm & Product & & Mean & Min & Max & SD & Mean & Min & Max & SD \\\\\n');
fprintf(fileID, '\\hline\n');
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

% Export the Table 
% Open a file to write the LaTeX table
fileID=fopen([outputPath,'regression_results_ols.tex'],'w');

% Write the LaTeX table
fprintf(fileID, '\\begin{tabular}{lcccc}\n');
fprintf(fileID, '\\hline\n');
fprintf(fileID, 'Variable & Coef & SE & CI$_{low}$ & CI$_{upp}$ \\\\\n');
fprintf(fileID, '\\hline\n');

for i = 1:height(results_table)
    fprintf(fileID, '%s & %.4f & %.4f & %.4f & %.4f \\\\\n', ...
        RowNames{i}, results_table{i, 'Coef'}, results_table{i, 'SE'}, ...
        results_table{i, 'CI_low'}, results_table{i, 'CI_upp'});
end

fprintf(fileID, '\\hline\n');
fprintf(fileID, '\\end{tabular}\n');

% Close the file
fclose(fileID);

disp('Regression results exported to "regression_results_ols.tex".');

%%%%%%%%%%%%%
% QUESTION 4
%%%%%%%%%%%%%
% BLP INSTRUMENT
% IN-BLP
% Initialize the new variable
blp_in = zeros(height(market), 1);

% Loop over each row to calculate the sum of quantities
for i = 1:height(market)

    % CURRENT FIRM PRODUCT MARKET
    current_firm = firm(i);
    current_market = market(i);
    current_product = product(i);
    
    % INDEX FOR PRODUCTS IN SAME FIRM-MARKET
    same_firm_market = (firm == current_firm) & (market == current_market);
    
    % EXCLUDE CURRENT PRODUCT
    exclude_self = product ~= current_product;
    
    % Calculate the sum of quantities, excluding the current product
    blp_in(i) = sum(x(same_firm_market & exclude_self));
end

% OUT-BLP
% Initialize the new variable
blp_out = zeros(height(market), 1);

% Loop over each row to calculate the sum of characteristics
for i = 1:height(market)

    % CURRENT FIRM AND MARKET
    current_firm = firm(i);
    current_market = market(i);
    
    % PRODUCTS WITHIN SAME MARKET BUT NOT SAME FIRM
    not_same_firm_market = (firm ~= current_firm) & (market == current_market);
    
    % Calculate the sum of characteristics for products not belonging to the firm
    blp_out(i) = sum(x(not_same_firm_market));
end

% GMM WITH COST SHIFTER ONLY
% AS THE MODEL IS JUST IDENTIFIED, THE DETERMINATION OF W IS IRRELEVANT
% WE TAKE THE WEIGHTING MATRIX THAT LEADS TO THE NL2SLS
Zexcluded_1=[v];
Z_d1=[ones(size(market,1),1) x Zexcluded_1];
W=(Z_d1'*Z_d1)\eye(size(Z_d1,2));
theta_d_GMM=(X_d'*Z_d1*W*Z_d1'*X_d)\X_d'*Z_d1*W*Z_d1'*delta_output;
xi_GMM=delta_output-X_d*theta_d_GMM;
S=Z_d1'*diag(xi_GMM.^2)*Z_d1;
VAR_d_GMM=(X_d'*Z_d1*W*(X_d'*Z_d1)')\X_d'*Z_d1*W*S*W*(X_d'*Z_d1)'/(X_d'*Z_d1*W*(X_d'*Z_d1)');
SE_d_GMM=sqrt(diag(VAR_d_GMM));
CIlow_d_GMM=theta_d_GMM-SE_d_GMM*1.96;
CIupp_d_GMM=theta_d_GMM+SE_d_GMM*1.96;

RowNames={'Constant','Beta','Alpha'};
VariableNames={'Coef','SE','CI_low','CI_upp'};
results_table=table(theta_d_GMM,SE_d_GMM,CIlow_d_GMM,CIupp_d_GMM,...
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

% GMM WITH BLP INSTRUMENTS ONLY
% AS WE HAVE TWO INSTRUMENTS FOR ONE ENDOGENEOUS PARAMETER, THE MODEL IS
% OVERIDENTIFIED
% WE PROCEED TO THE TWO STEP METHOD TO GET THE OPTIMAL WEIGHTING MATRIX

% FIRST STEP
Zexcluded_2=[blp_in blp_out];
Z_d2=[Zexcluded_2 ones(size(market,1),1) x];
W=(Z_d2'*Z_d2)\eye(size(Z_d2,2));
theta_GMM2=(X_d'*Z_d2*W*Z_d2'*X_d)\X_d'*Z_d2*W*Z_d2'*delta_output;
xi_GMM2=delta_output-X_d*theta_GMM2;

% SECOND STEP
W              = (Z_d2' * diag(xi_GMM2.^2) * Z_d2) \ eye(size(Z_d2,2));
theta_GMM2 = (X_d' * Z_d2 * W * Z_d2' * X_d) \ X_d' * Z_d2 * W * Z_d2' * delta_output;
xi_GMM2    = delta_output - X_d * theta_GMM2;
S              = Z_d2' * diag(xi_GMM2.^2) * Z_d2;
VAR_GMM2   = ( X_d' * Z_d2 * W * (X_d' * Z_d2)' ) \ X_d' * Z_d2 * W * S * W * (X_d' * Z_d2)' / ( X_d' * Z_d2 * W * (X_d' * Z_d2)' ); 
SE_GMM2    = sqrt(diag(VAR_GMM2));
CIlow_GMM2 = theta_GMM2 - SE_GMM2 * 1.96;
CIupp_GMM2 = theta_GMM2 + SE_GMM2 * 1.96;
table(theta_GMM2,SE_GMM2,CIlow_GMM2,CIupp_GMM2,'RowNames',{'constante','beta','alpha'},'VariableNames',{'Coef' 'SE' 'CI_low' 'CI_upp'})

RowNames={'Constant','Beta','Alpha'};
VariableNames={'Coef','SE','CI_low','CI_upp'};
results_table=table(theta_GMM2,SE_GMM2,CIlow_GMM2,CIupp_GMM2,...
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

% GMM WITH ALL OF THE INSTRUMENTS
Zexcluded_3=[blp_in blp_out v];
Z_d3=[ones(size(market,1),1) x Zexcluded_1 Zexcluded_2];

W=(Z_d3'*Z_d3)\eye(size(Z_d3,2));
theta_GMM3=(X_d'*Z_d3*W*Z_d3'*X_d)\X_d'*Z_d3*W*Z_d3'*delta_output;
xi_GMM3=delta_output-X_d*theta_GMM3;

% SECOND STEP
W              = (Z_d3' * diag(xi_GMM3.^2) * Z_d3) \ eye(size(Z_d3,2));
theta_GMM3 = (X_d' * Z_d3 * W * Z_d3' * X_d) \ X_d' * Z_d3 * W * Z_d3' * delta_output;
xi_GMM3    = delta_output - X_d * theta_GMM3;
S              = Z_d3' * diag(xi_GMM3.^2) * Z_d3;
VAR_GMM3   = ( X_d' * Z_d3 * W * (X_d' * Z_d3)' ) \ X_d' * Z_d3 * W * S * W * (X_d' * Z_d3)' / ( X_d' * Z_d3 * W * (X_d' * Z_d3)' ); 
SE_GMM3    = sqrt(diag(VAR_GMM3));
CIlow_GMM3 = theta_GMM3 - SE_GMM3 * 1.96;
CIupp_GMM3 = theta_GMM3 + SE_GMM3 * 1.96;
table(theta_GMM3,SE_GMM3,CIlow_GMM3,CIupp_GMM3,'RowNames',{'constante','beta','alpha'},'VariableNames',{'Coef' 'SE' 'CI_low' 'CI_upp'})

RowNames={'Constant','Beta','Alpha'};
VariableNames={'Coef','SE','CI_low','CI_upp'};
results_table=table(theta_GMM3,SE_GMM3,CIlow_GMM3,CIupp_GMM3,...
    'RowNames',RowNames,'VariableNames',VariableNames);

fileID=fopen([outputPath,'gmm_results_all.tex'],'w');
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


% WEAK INSTRUMENT TEST: F-stat (case 1)
theta_price_1  = (Z_d1'*Z_d1) \ Z_d1' * price; % Regress the endogeneous variable p on instruments.
res_price_1    = price - Z_d1 * theta_price_1;
VARCOV_price_1 = ((res_price_1' * res_price_1) / (size(Z_d1,1) - size(Z_d1,2)) ) * ((Z_d1'*Z_d1) \ eye(size(Z_d1,2)));
H = zeros(size(Z_d1,2),size(Z_d1,2));
i = 1;
for j = 1 : size(Zexcluded_1,2)
    H(i,j) = 1;
    i      = i + 1;
end
c = zeros(size(Z_d1,2),1);
[p,F_stat_1]=linhyptest(theta_price_1,VARCOV_price_1,c,H,size(Z_d1,1)-size(Z_d1,2));
display(sprintf('\n 1st F-stat: %f %d',F_stat_1,p)) 
clear theta_price_1 res_price_1 VARCOV_price_1 se_price_1 T_test_price_1 pvalue_price_1  H i c p

% WEAK INSTRUMENT TEST: F-stat (case 2)
theta_price_2  = (Z_d2'*Z_d2) \ Z_d2' * price; % Regress the endogeneous variable p on instruments.
res_price_2    = price - Z_d2 * theta_price_2;
VARCOV_price_2 = ((res_price_2' * res_price_2) / (size(Z_d2,1) - size(Z_d2,2)) ) * ((Z_d2'*Z_d2) \ eye(size(Z_d2,2)));
H = zeros(size(Z_d2,2),size(Z_d2,2));
i = 1;
for j = 1 : size(Zexcluded_2,2)
    H(i,j) = 1;
    i      = i + 1;
end
c = zeros(size(Z_d2,2),1);
[p,F_stat_2]=linhyptest(theta_price_2,VARCOV_price_2,c,H,size(Z_d2,1)-size(Z_d2,2));
display(sprintf('\n 1st F-stat: %f %d',F_stat_2,p)) 
clear theta_price_2 res_price_2 VARCOV_price_2 se_price_2 T_test_price_2 pvalue_price_2  H i c p

% WEAK INSTRUMENT TEST: F-stat (case 3)
theta_price_3  = (Z_d3'*Z_d3) \ Z_d3' * price; % Regress the endogeneous variable p on instruments.
res_price_3    = price - Z_d3 * theta_price_3;
VARCOV_price_3 = ((res_price_3' * res_price_3) / (size(Z_d3,1) - size(Z_d3,2)) ) * ((Z_d3'*Z_d3) \ eye(size(Z_d3,2)));
H = zeros(size(Z_d3,2),size(Z_d3,2));
i = 1;
for j = 1 : size(Zexcluded_3,2)
    H(i,j) = 1;
    i      = i + 1;
end
c = zeros(size(Z_d3,2),1);
[p,F_stat_3]=linhyptest(theta_price_3,VARCOV_price_3,c,H,size(Z_d3,1)-size(Z_d3,2));
display(sprintf('\n 1st F-stat: %f %d',F_stat_3,p)) 
clear theta_price_3 res_price_3 VARCOV_price_3 se_price_3 T_test_price_3 pvalue_price_3  H i c p 

% PERFORMING HANSEN TEST OF OVER-IDENTIFICATION
% Case 3
% H0: E[Z_d*\xi]=0 (test the orthogonality conditions) %
W = (Z_d3' * diag(xi_GMM3.^2) * Z_d3) \ eye(size(Z_d3,2));
if size(Z_d3,2)-(size(X_d,2)+1) > 0
    OIR       = xi_GMM3' * Z_d3 * W * (xi_GMM3' * Z_d3)';
    pvalueOIR = 1 - chi2cdf(OIR,size(Z_d3,2)-(size(X_d,2)+1));
    table(OIR,pvalueOIR,'RowNames',{'Hansen_test'},'VariableNames',{'Stat','pvalue'})
end

% We keep the just identified restriction (and assume that the
% excludability holds)

%%%%%%%%%%%%%%%%%%%%%
% SUPPLY - QUESTION 5
%%%%%%%%%%%%%%%%%%%%%
% MARKET SHARE ESTIMATED FROM LOGIT
nume=zeros(1,J);
deno=zeros(1,1);
deno=exp(deno);
market_share_hat_matrix=zeros(1,J);
market_share_hat=zeros(1*J,1);

% Loop over markets and products
for t=1
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

% ESTIMATED JACOBIAN DEMAND
Qp=zeros(J,J);
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

%%%%%%%%%%%%%
% QUESTION 6
%%%%%%%%%%%%%
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
saveas(gcf,[outputPath,'mc_distribution.png']);

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
    price_merge_approx(market==t,1)= mc_hat_comp(market==t,1)-(Qp(market==t,:).*OF_merge)\market_share_hat(market==t,1);
end


% Compute percentage price change
price_change_comp = (price_merge_approx - price) ./ price * 100;

% Compute the average percentage price change for firm 3 & 4 together
firms_3_4_indices = ismember(firm, [3, 4]); % Logical indices for firms 3 & 4
average_price_change_firms_3_4 = mean(price_change_comp(firms_3_4_indices)); % Average with NaN ignored

% Compute the average percentage price change for firm 1 & 2 together
firms_1_2_indices = ismember(firm, [1, 2]); % Logical indices for firms 1 & 2
average_price_change_firms_1_2 = mean(price_change_comp(firms_1_2_indices)); % Average with NaN ignored

% Display the results for competition
disp('Average Percentage Price Change in competition case:');
disp(['Firms 3 & 4: ', num2str(average_price_change_firms_3_4), '%']);
disp(['Firms 1 & 2: ', num2str(average_price_change_firms_1_2), '%']);


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
% Define a global variable to store the history
global history;
history = []; % Initialize an empty array to store parameter history

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
        'Display', 'final-detailed',...
        'OutputFcn', @outfun);   % Display iteration information
%     price_merge2(J*(t-1)+1:J*t,1)  = fsolve(@(price_post) mergerSystem(price_post,x,theta_d_GMM,xi_GMM,t,J,market,OF_merge,mc_hat_comp,nume,deno,market_share_hat,market_share_hat_matrix,Qp), price_0, options);
    price_merge2(J*(t-1)+1:J*t,1)  = fsolve(@(price_post) mergerSystem2(price_post,x,theta_d_GMM,xi_GMM,t,J,market,OF_merge,mc_hat_comp,nume,deno,market_share_hat,market_share_hat_matrix,Qp), price_0, options);
end
histogram(price_merge2)
% pb: some prices negatives and overall too low
plot(history(1:500));

%%%%%%%%%%%%%
% QUESTION 9
%%%%%%%%%%%%%
 clearvars -except outputPath
 clc
 load ('Data_exam_post')

% Define merging and non-merging firms
merging_firms = [3, 4]; % Adjust these IDs to the merging firms
non_merging_firms = [1, 2]; % All other firms

% Get the unique markets
markets = unique(market);

% Initialize vectors to store average prices
avg_price_merging = zeros(length(markets), 1);
avg_price_non_merging = zeros(length(markets), 1);

% Loop through each market
for t = 1:length(markets)
    current_market = markets(t);
    
    % Logical indices for the current market
    market_indices = (market == current_market);
    
    % Prices for merging firms in this market
    merging_prices = price(market_indices & ismember(firm, merging_firms));
    avg_price_merging(t) = mean(merging_prices); % Average price for merging firms
    
    % Prices for non-merging firms in this market
    non_merging_prices = price(market_indices & ismember(firm, non_merging_firms));
    avg_price_non_merging(t) = mean(non_merging_prices); % Average price for non-merging firms
end

% Plot the trends
figure;
plot(markets, avg_price_merging, '-o', 'LineWidth', 1.5, 'DisplayName', 'Merging Firms');
hold on;
plot(markets, avg_price_non_merging, '-s', 'LineWidth', 1.5, 'DisplayName', 'Non-Merging Firms');

% Add vertical line for merging time (x = 150)
xline(150, '--r', 'Merging time', 'LineWidth', 1.5, ...
    'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'center');

hold off;

% Customize the plot
title('Average Price Trends: Merging vs. Non-Merging Firms');
xlabel('Market');
ylabel('Average Price');
legend('Location', 'Best');
grid on;

saveas(gcf,[outputPath,'Average_Price_Trends.png']);

%%%%%%%%%%%%%
% QUESTION 10
%%%%%%%%%%%%%
merging_dummy = ismember(firm, merging_firms); 

% TREND
Trend = market; 

% LOG-PRICES
log_price = log(price);

% INTERACTION
Trend_merging = Trend .* merging_dummy;

% PRODUCT FE
product_dummies = customDummyVar(product); 
% Restrict data to pre-merger period (first 3000 observations)
pre_merger_indices = 1:3000; % Indices for pre-merger data
log_price_pre = log_price(pre_merger_indices); % Pre-merger log prices
Trend_pre = Trend(pre_merger_indices); % Pre-merger trend
merging_dummy_pre = merging_dummy(pre_merger_indices); % Pre-merger dummy variable
product_dummies_pre = product_dummies(pre_merger_indices, :); % Pre-merger product dummies

% Interaction term for pre-merger data
Trend_merging_pre = Trend_pre .* merging_dummy_pre;

% Design matrix (excluding one product dummy for multicollinearity)
X_d_pre = [Trend_pre, Trend_merging_pre, product_dummies_pre(:, 2:end)]; 

% Estimate the coefficients using OLS for pre-merger data
theta_MNL_OLS_pre = (X_d_pre' * X_d_pre) \ (X_d_pre' * log_price_pre); % OLS formula
res_MNL_OLS_pre = log_price_pre - X_d_pre * theta_MNL_OLS_pre; % Compute residuals

% Variance-Covariance Matrix
VAR_MNL_OLS_pre = ((res_MNL_OLS_pre' * res_MNL_OLS_pre) / (size(X_d_pre, 1) - size(X_d_pre, 2))) * ...
    ((X_d_pre' * X_d_pre) \ eye(size(X_d_pre, 2))); 
SE_MNL_OLS_pre = sqrt(diag(VAR_MNL_OLS_pre)); % Standard errors
CIlow_MNL_OLS_pre = theta_MNL_OLS_pre - SE_MNL_OLS_pre * 1.96; % 95% CI lower bound
CIupp_MNL_OLS_pre = theta_MNL_OLS_pre + SE_MNL_OLS_pre * 1.96; % 95% CI upper bound

% Focus on the coefficients of interest: Trend and Interaction
theta_interest_pre = theta_MNL_OLS_pre(1:2); % First two coefficients (Trend, Interaction)
SE_interest_pre = SE_MNL_OLS_pre(1:2);
CIlow_interest_pre = CIlow_MNL_OLS_pre(1:2);
CIupp_interest_pre = CIupp_MNL_OLS_pre(1:2);

% Table content for Trend and Interaction only (pre-merger)
RowNames = {'Trend', 'Interaction'};
VariableNames = {'Coef', 'SE', 'CI_low', 'CI_upp'};
results_table_pre = table(theta_interest_pre, SE_interest_pre, CIlow_interest_pre, CIupp_interest_pre, ...
    'RowNames', RowNames, 'VariableNames', VariableNames);

% Export the Table 
fileID=fopen([outputPath,'pretrend_results_ols.tex'],'w');

% Write the LaTeX table
fprintf(fileID, '\\begin{tabular}{lcccc}\n');
fprintf(fileID, '\\hline\n');
fprintf(fileID, 'Variable & Coef & SE & CI$_{low}$ & CI$_{upp}$ \\\\\n');
fprintf(fileID, '\\hline\n');

for i = 1:height(results_table_pre)
    fprintf(fileID, '%s & %.4f & %.4f & %.4f & %.4f \\\\\n', ...
        RowNames{i}, results_table_pre{i, 'Coef'}, results_table_pre{i, 'SE'}, ...
        results_table_pre{i, 'CI_low'}, results_table_pre{i, 'CI_upp'});
end

fprintf(fileID, '\\hline\n');
fprintf(fileID, '\\end{tabular}\n');

% Close the file
fclose(fileID);


%%%%%%%%%%%%%
% QUESTION 11
%%%%%%%%%%%%%
% Create the Post dummy variable
% Define the interaction term directly
Post = market > T_post;
Post_Merging = Post .* merging_dummy; % Interaction term for Post and MergingFirm

% Design matrix with the interaction term, ProductFE, and MarketFE
X_d = [Post_Merging, product_dummies(:, 2:end), market(:, 2:end)]; 

% Estimate coefficients using OLS
theta_simple = (X_d' * X_d) \ (X_d' * log_price); % OLS formula
res_simple = log_price - X_d * theta_simple; % Compute residuals

% Variance-Covariance Matrix
VAR_simple = ((res_simple' * res_simple) / (size(X_d, 1) - size(X_d, 2))) * ...
    ((X_d' * X_d) \ eye(size(X_d, 2)));
SE_simple = sqrt(diag(VAR_simple)); % Standard errors

% Confidence intervals
CIlow_simple = theta_simple - SE_simple * 1.96; % 95% CI lower bound
CIupp_simple = theta_simple + SE_simple * 1.96; % 95% CI upper bound

% Extract the coefficient, SE, and CI for the interaction term (Post_Merging)
beta_interaction = theta_simple(1); % First coefficient corresponds to Post_Merging
SE_interaction = SE_simple(1); % Standard error for Post_Merging
CI_interaction_low = CIlow_simple(1); % CI lower bound for Post_Merging
CI_interaction_upp = CIupp_simple(1); % CI upper bound for Post_Merging

% Create the results table for the interaction term
RowNames = {'Post_Merging'};
VariableNames = {'Coef', 'SE', 'CI_low', 'CI_upp'};
results_table_simple = table(beta_interaction, SE_interaction, CI_interaction_low, CI_interaction_upp, ...
    'RowNames', RowNames, 'VariableNames', VariableNames);

% Export the Table to a LaTeX file
fileID=fopen([outputPath,'interaction_results.tex'],'w');

% Write the LaTeX table
fprintf(fileID, '\\begin{tabular}{lcccc}\n');
fprintf(fileID, '\\hline\n');
fprintf(fileID, 'Variable & Coef & SE & CI$_{low}$ & CI$_{upp}$ \\\\\n');
fprintf(fileID, '\\hline\n');
fprintf(fileID, '%s & %.4f & %.4f & %.4f & %.4f \\\\\n', ...
    RowNames{1}, results_table_simple{1, 'Coef'}, results_table_simple{1, 'SE'}, ...
    results_table_simple{1, 'CI_low'}, results_table_simple{1, 'CI_upp'});
fprintf(fileID, '\\hline\n');
fprintf(fileID, '\\end{tabular}\n');

% Close the file
fclose(fileID);

disp('Regression results for interaction term exported to "interaction_results.tex".');
