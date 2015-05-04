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
%First manually make entries in dataset 0 - unlabelled

positive_indices = find(true_labels==1);
A = M(positive_indices,:);

negative_indices = find(true_labels==-1);
B = M(negative_indices,:);

max_accuracy = 0.0;
best_c_1 = 0.0;
best_c_2 = 0.0;
best_c_3 = 0.0;
best_sigma = 0.0;
best_k = 0;
best_predicted = zeros(size(y));

for c_1 = 2.^[-1:1]
    for c_2 = 2.^[-1:1]
        for c_3 = 2.^[-1:1]
            for sigma = 2.^[-1:1]
                for k = [1:2]

                    fprintf('c_1: %f, c_2: %f, c_3: %f, sigma: %f, k: %f',c_1,c_2,c_3,sigma,k);
                    
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
                    accuracy =  sum(correct)/size(correct,1);
                    
                    fprintf('Accuracy: %f\n\n',accuracy);
                    
                    if(accuracy>max_accuracy)
                       max_accuracy = accuracy;
                       best_c_1 = c_1;
                       best_c_2 = c_2;
                       best_c_3 = c_3;
                       best_sigma = sigma;
                       best_k = k;
                    end
                end
            end
        end
    end
end

fprintf('Best Accuracy: %f\n',max_accuracy);
fprintf('Best c_1: %f\n',best_c_1);
fprintf('Best c_2: %f\n',best_c_2);
fprintf('Best c_3: %f\n',best_c_3);
fprintf('Best sigma: %f\n',best_sigma);
fprintf('Best k: %f\n',best_k);

% Plot
figure;
scatter(x(:,1),x(:,2),[],best_predicted);
hold on;
scatter(x(pos_class_pt,1),x(pos_class_pt,2),[],'r');
scatter(x(neg_class_pt,1),x(neg_class_pt,2),[],'g');