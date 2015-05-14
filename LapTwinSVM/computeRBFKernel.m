function [K] = computeRBFKernel(A,B,sigma)
    K = exp(-(pdist2(A,B).^2)./(2*sigma^2));
end