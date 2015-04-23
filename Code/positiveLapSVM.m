function [rho_plus] = positiveLapSVM(L,M,c_1,c_2,c_3,n_positive,n_negative,n_unlabelled,sigma)
    n_labelled = n_positive + n_negative;
    total      = n_labelled + n_unlabelled;
    
    O_phi   = [horzcat(computeRBFKernel(M,M,sigma),zeros(total,1));[zeros(1,total) 1]]; 
    e       = ones(total,1);      %(l+u)
    J_phi   = horzcat(computeRBFKernel(M,M,sigma), e); 
    e_plus  = ones(n_positive,1); % m1
    e_minus = ones(n_negative,1); % m2
    H_phi   = horzcat(computeRBFKernel(A,M,sigma) ,e_plus);
    G_phi   = horzcat(computeRBFKernel(B,M,sigma),e_minus);

    H = G_phi* inv(H_phi'*H_phi + c_2*O_phi+ c_3*J_phi'*L*J_phi)*G_phi';

    lb  = zeros(n,1);   % lower bound for alpha
    ub  = c_1* e_minus; % upper bound for beta
    
    f = e_minus;
    alpha = quadprog(H,f,A,b,[],[],lb,ub);

    %rho_plus = [lamda_plus b_plus]^T
    rho_plus = -1* inv(H_phi'*H_phi + c_2*O_phi+c_3*J_phi'*L*J_phi)*G_phi'*alpha;
end

