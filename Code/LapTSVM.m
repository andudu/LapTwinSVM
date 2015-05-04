%load('2moon.mat');
%pos_class_pt = 123;
%neg_class_pt = 142;

load('clock.mat');
pos_class_pt = 127;
neg_class_pt = 1;

M = x;
true_labels = zeros(size(y));
true_labels(pos_class_pt) = 1;
true_labels(neg_class_pt) = -1;

total = size(M,1);
%1:positive
%-1: negative
%First manually make entries in dataset 0- unlabelled

positive_indices = find(true_labels==1);
A = M(positive_indices,:);

negative_indices = find(true_labels==-1);
B = M(negative_indices,:);

% Parameters
c_1 = 2;
c_2 = 2;
c_3 = 2;
sigma = 0.5;
k = 9;

% Calculate L using D,W (First we need to find W)
IDX = knnsearch(M,M,'K',k);
W = zeros(total,total);
D = zeros(total,total);
for i = 1:total
   for j = 1:k
       val = exp(-norm(M(i,:)-M(IDX(i,j),:))/(2*sigma^2));
       W(i,IDX(i,j)) = val;
       W(IDX(i,j),i) = val;
   end
   D(i,i) = sum(W(i,:));
end
L = D-W;

positiveRes = positiveLapSVM(L,A,B, M,c_1,c_2,c_3,sigma);
negativeRes = negativeLapSVM(L,A,B, M,c_1,c_2,c_3,sigma);

lambda_plus = positiveRes(1:total,:);
b_plus = positiveRes(total+1,:);

lambda_minus = negativeRes(1:total,:);
b_minus = negativeRes(total+1,:);

e = zeros(total,1);
f_plus = computeRBFKernel(M,M,sigma)*lambda_plus + e*b_plus;
f_minus = computeRBFKernel(M,M,sigma)*lambda_minus + e*b_minus;

%Find minimum of the distance to two hyperplanes and then classify to
%positive or negative. Find accuracy

predicted = 2*(min(abs(f_plus),abs(f_minus)) == abs(f_plus))-1;
correct = (predicted == y);
fprintf('Accuracy: %f\n',sum(correct)/size(correct,1));

% Plot
figure;
scatter(x(:,1),x(:,2),[],predicted);
hold on;
scatter(x(pos_class_pt,1),x(pos_class_pt,2),[],'r');
scatter(x(neg_class_pt,1),x(neg_class_pt,2),[],'g');