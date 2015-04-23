function [rho_minus ] = negativeLapSVM(L,A,B,M,c_1,c_2,c_3,n_positive,n_negative,n_unlabelled)
n_labelled= n_positive + n_negative;
total = n_labelled + n_unlabelled;
U_phi = [horzcat(K(M,M),zeros(total,1));[zeros(1,total) 1]];
e=ones(total,1); %(l+u)
F_phi =horzcat(K(M,M), e);
e_plus = ones(n_positive,1); %m1
e_minus = ones(n_negative,1); % m2
Q_phi = horzcat(K(A,M') ,e_minus);
P_phi =  horzcat(K(B,M'),e_plus);

H = -1*P_phi* inv(Q_phi'*Q_phi + c_2*U_phi+ c_3*F_phi'*L*F_phi)*P_phi';

lb= zeros(n_positive,1); % lower bound for alpha
ub = c_2* e_plus; % upper bound for beta
f = e_plus ;
beta = quadprog(H,f,[],[],[],[],lb,ub);

%rho_plus = [lamda_minus b_minus]^T
rho_minus=-1* inv(Q_phi'*Q_phi + c_2*U_phi+c_3*F_phi'*L*F_phi)*P_phi'*beta;
end
