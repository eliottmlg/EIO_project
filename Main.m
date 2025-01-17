clc; clear; close all;
cd('C:\Users\eliot\Documents\REPOSITORIES\EIO_project');
addpath('C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Data\');
addpath('C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Utils\');
addpath('C:\Users\eliot\Documents\REPOSITORIES\EIO_project\Outputs\');


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

fileID=fopen('C:\Users\eliot\Documents\REPOSITORIES\EIO_project/Outputs/summary_table.tex','w');
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
clear;
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

fileID=fopen('C:\Users\eliot\Documents\REPOSITORIES\EIO_project/Outputs/regression_results_ols.tex','w');
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

fileID=fopen('C:\Users\eliot\Documents\REPOSITORIES\EIO_project/Outputs/gmm_results_cost.tex','w');
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
S=Z_d'*diag(xi_GMM.^2)*Z_d;
VAR_d_GMM=(X_d'*Z_d*W*(X_d'*Z_d)')\X_d'*Z_d*W*S*W*(X_d'*Z_d)'/(X_d'*Z_d*W*(X_d'*Z_d)');
SE_d_GMM=sqrt(diag(VAR_d_GMM));
CIlow_d_GMM=theta_d_GMM2-SE_d_GMM*1.96;
CIupp_d_GMM=theta_d_GMM2+SE_d_GMM*1.96;

% PERFORMING HANSEN TEST OF OVER-IDENTIFICATION
hansenTestGMM(delta_output, X_d, Z_d, theta_d_GMM)

RowNames={'Constant','Beta','Alpha'};
VariableNames={'Coef','SE','CI_low','CI_upp'};
results_table=table(theta_d_GMM2,SE_d_GMM,CIlow_d_GMM,CIupp_d_GMM,...
    'RowNames',RowNames,'VariableNames',VariableNames);

fileID=fopen('C:\Users\eliot\Documents\REPOSITORIES\EIO_project/Outputs/gmm_results_blp.tex','w');
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
S=Z_d'*diag(xi_GMM.^2)*Z_d;
VAR_d_GMM=(X_d'*Z_d*W*(X_d'*Z_d)')\X_d'*Z_d*W*S*W*(X_d'*Z_d)'/(X_d'*Z_d*W*(X_d'*Z_d)');
SE_d_GMM=sqrt(diag(VAR_d_GMM));
CIlow_d_GMM=theta_d_GMM3-SE_d_GMM*1.96;
CIupp_d_GMM=theta_d_GMM3+SE_d_GMM*1.96;

% PERFORMING HANSEN TEST OF OVER-IDENTIFICATION
hansenTestGMM(delta_output, X_d, Z_d, theta_d_GMM3)

RowNames={'Constant','Beta','Alpha'};
VariableNames={'Coef','SE','CI_low','CI_upp'};
results_table=table(theta_d_GMM3,SE_d_GMM,CIlow_d_GMM,CIupp_d_GMM,...
    'RowNames',RowNames,'VariableNames',VariableNames);

fileID=fopen('C:\Users\eliot\Documents\REPOSITORIES\EIO_project/Outputs/gmm_results_cost_blp.tex','w');
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
        utility=theta_d_GMM3(1,1)+theta_d_GMM3(2,1)*x((t-1)*J+j,1)+theta_d_GMM3(3,1)*price((t-1)*J+j,1)+xi_GMM3((t-1)*J+j,1);
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
        Qp(market==t,j)=abs(theta_d_GMM3(3,1))*(market_share_hat(market==t,1).*market_share_hat_matrix(t,j));
    end
    Qp(market==t,:)=Qp(market==t,:).*(ones(J,J)-eye(J));
    Qp(market==t,:)=Qp(market==t,:)+diag(theta_d_GMM3(3,1)*(market_share_hat(market==t,1).*(ones(J,1)-market_share_hat(market==t,1))));
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

% PRICE-COST MARGIN UNDER COLLUSION of FIRMS 3 and 4
markup_hat_partialcol=zeros((J*T_pre),1);
for t=1:T_pre
    markup_hat_partialcol(market==t,1)=-(Qp(market==t,:).*OFcoll)\market_share_hat(market==t,1);
end

% HISTOGRAM OF PRICE COST MARGINS
figure;
subplot(1,3,1);
histogram(markup_hat_compet,'BinWidth',0.01,'FaceAlpha',0.7,'EdgeColor','none');
xlabel('Markup (Competition)','FontSize',8);
ylabel('Count','FontSize',8);
grid on;

subplot(1,3,2);
histogram(markup_hat_partialcol,'BinWidth',0.01,'FaceAlpha',0.7,'EdgeColor','none');
xlabel('Markup (Partial collusion, only 3 and 4)','FontSize',8);
ylabel('Count','FontSize',8);
grid on;

subplot(1,3,3);
histogram(markup_hat_col,'BinWidth',0.01,'FaceAlpha',0.7,'EdgeColor','none');
xlabel('Markup (Full collusion)','FontSize',8);
ylabel('Count','FontSize',8);

% MARGINAL COST UNDER COMPETITION
mc_hat_comp=zeros((J*T_pre),1);
for t=1:T_pre
mc_hat_comp(market==t,1) = price(market==t,1) + (Qp(market==t,:).*OF) \ market_share_hat(market==t,1);
end

% MARGINAL COST UNDER COMPLETE COLLUSION
mc_hat_col=zeros((J*T_pre),1);
for t=1:T_pre
mc_hat_col(market==t,1) = price(market==t,1) + (Qp(market==t,:)) \ market_share_hat(market==t,1);
end

% MARGINAL COST UNDER COLLUSION OF 3 AND 4
mc_hat_partialcol=zeros((J*T_pre),1);
for t=1:T_pre
mc_hat_partialcol(market==t,1) = price(market==t,1) + (Qp(market==t,:).*OFcoll) \ market_share_hat(market==t,1);
end

% HISTOGRAMS OF MARGINAL COST
figure;
subplot(1,3,1);
histogram(mc_hat_comp,'BinWidth',0.01,'FaceAlpha',0.7,'EdgeColor','none');
xlabel('Marginal costs (Competition)','FontSize',8);
ylabel('Count','FontSize',8);
grid on;

subplot(1,3,2);
histogram(mc_hat_partialcol,'BinWidth',0.01,'FaceAlpha',0.7,'EdgeColor','none');
xlabel('Marginal costs (Partial collusion, only 3 and 4)','FontSize',8);
ylabel('Count','FontSize',8);
grid on;

subplot(1,3,3);
histogram(mc_hat_col,'BinWidth',0.01,'FaceAlpha',0.7,'EdgeColor','none');
xlabel('Marginal costs (Full collusion)','FontSize',8);
ylabel('Count','FontSize',8);



% HISTOGRAM OF PRICE COST MARGINS UNDER COLLUSION AND COMPETITION
unique_products=unique(product);
num_products=length(unique_products);

% Competition Histogram
close all
histogram(markup_hat_compet)
histogram(markup_hat_partialcol)

figure;
for i=1:num_products
    product_id=unique_products(i);
    markups_compet_product=markup_hat_compet(product==product_id);
    subplot(4,5,i);
    histogram(markups_compet_product,'BinWidth',0.01,'FaceAlpha',0.7,'EdgeColor','none');
    title(['Product ',num2str(product_id)],'FontSize',10);
    xlabel('Markup (Competition)','FontSize',8);
    ylabel('Count','FontSize',8);
    grid on;
end
sgtitle('Markup Distribution by Product (Competition)','FontSize',14,'FontWeight','bold');
saveas(gcf,'C:\Users\eliot\Documents\REPOSITORIES\EIO_project/Outputs/markup_distribution_competition.png');

% Collusion Histogram
figure;
for i=1:num_products
    product_id=unique_products(i);
    markups_coll_product=markup_hat_col(product==product_id);
    subplot(4,5,i);
    histogram(markups_coll_product,'BinWidth',0.01,'FaceAlpha',0.7,'EdgeColor','none');
    title(['Product ',num2str(product_id)],'FontSize',10);
    xlabel('Markup (Collusion)','FontSize',8);
    ylabel('Count','FontSize',8);
    grid on;
end
sgtitle('Markup Distribution by Product (Collusion)','FontSize',14,'FontWeight','bold');
saveas(gcf,'C:\Users\eliot\Documents\REPOSITORIES\EIO_project/Outputs/markup_distribution_collusion.png');
