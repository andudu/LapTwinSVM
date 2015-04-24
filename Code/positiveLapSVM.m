function [rho_plus ] = positiveLapSVM(L,A,B,M,c_1,c_2,c_3,sigma)

    n_positive = size(A,1);
    n_negative = size(B,1);
    total = size(M,1);

    O_phi = [horzcat(computeRBFKernel(M,M,sigma),zeros(total,1));[zeros(1,total) 1]];
    e = ones(total,1); %(l+u)
    J_phi = horzcat(computeRBFKernel(M,M,sigma), e);
    e_plus = ones(n_positive,1); %m1
    e_minus = ones(n_negative,1); % m2
    H_phi = horzcat(computeRBFKernel(A,M,sigma) ,e_plus);
    G_phi =  horzcat(computeRBFKernel(B,M,sigma),e_minus);

    H = -1*G_phi* inv(H_phi'*H_phi + c_2*O_phi+ c_3*J_phi'*L*J_phi)*G_phi';

    lb= zeros(n_negative,1); % lower bound for alpha
    ub = c_1* e_minus; % upper bound for beta
    f = e_minus;
    alpha = quadprog(-H,-f,[],[],[],[],lb,ub);

    %rho_plus = [lamda_plus b_plus]^T
    rho_plus =-1* inv(H_phi'*H_phi + c_2*O_phi+c_3*J_phi'*L*J_phi)*G_phi'*alpha;
end


