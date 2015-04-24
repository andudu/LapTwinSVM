function [rho_minus] = negativeLapSVM(L,A,B,M,c_1,c_2,c_3,sigma)
    n_positive = size(A,1);
    n_negative = size(B,1);
    total = size(M,1);
    
    U_phi = [horzcat(computeRBFKernel(M,M,sigma),zeros(total,1));[zeros(1,total) 1]];
    e=ones(total,1); %(l+u)
    F_phi =horzcat(computeRBFKernel(M,M,sigma), e);
    e_plus = ones(n_positive,1); %m1
    e_minus = ones(n_negative,1); % m2
    Q_phi = horzcat(computeRBFKernel(A,M,sigma) ,e_minus);
    P_phi =  horzcat(computeRBFKernel(B,M,sigma),e_plus);

    H = -1*P_phi* inv(Q_phi'*Q_phi + c_2*U_phi+ c_3*F_phi'*L*F_phi)*P_phi';

    lb= zeros(n_positive,1); % lower bound for alpha
    ub = c_2* e_plus; % upper bound for beta
    f = e_plus ;
    beta = quadprog(H,f,[],[],[],[],lb,ub);

    %rho_plus = [lamda_minus b_minus]^T
    rho_minus=-1* inv(Q_phi'*Q_phi + c_2*U_phi+c_3*F_phi'*L*F_phi)*P_phi'*beta;
end
