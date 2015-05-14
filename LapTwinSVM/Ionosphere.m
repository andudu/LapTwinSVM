% Accuracy is on TEST set
% Number of rows = number of examples
% Number of columns = attributes
% Last column - Class Label

% Train, test set choosen randomly- Same with labelled and unlabelled set.

data = csvread('ionosphere.dat');

x= data(:,1:size(data,2)-1);
y= data(:,size(data,2));

positive_indices = find(y==1);
negative_indices = find(y==-1);

positive_data = x(positive_indices,:);
negative_data = x(negative_indices,:);

temp = randperm(size(positive_indices,1));
training_positive = positive_data(temp(1:0.65*size(positive_indices)),:);
test_positive = positive_data(temp(0.65*size(positive_indices):size(positive_indices)),:);
temp = randperm(size(training_positive,1));
labelled_positive = training_positive(temp(1:0.35*size(positive_indices)),:);
unlabelled_positive = training_positive(temp(0.35*size(positive_indices):size(training_positive)),:);

temp = randperm(size(negative_indices,1));
training_negative = negative_data(temp(1:0.65*size(negative_indices)),:);
test_negative = negative_data(temp(0.65*size(negative_indices):size(negative_indices)),:);
temp = randperm(size(training_negative,1));
labelled_negative = training_negative(temp(1:0.35*size(negative_indices)),:);
unlabelled_negative = training_negative(temp(0.35*size(negative_indices):size(training_negative)),:);

M = [labelled_positive; labelled_negative; unlabelled_positive; unlabelled_negative];
true_labels = zeros(size(M));
true_labels(1:size(labelled_positive)) = 1;
true_labels(size(labelled_positive)+1:size(labelled_positive)+size(labelled_negative)) = -1;
 
total = size(M,1);
A = labelled_positive;
B = labelled_negative;
z = [ ones(size(labelled_positive,1),1); ones(size(labelled_negative,1),1)*-1; ones(size(unlabelled_positive,1),1); ones(size(unlabelled_negative,1),1)*-1];

max_accuracy = 0.0;
best_c_1 = 0.0;
best_c_2 = 0.0;
best_c_3 = 0.0;
best_sigma = 0.0;
best_k = 0;
best_predicted = zeros(size(y));

% Vary in required range.
for c_1 =  2.^[-1:5]
    for c_2 =  2.^[-1:5]
        for c_3 =  2.^[-1:5]
            for sigma = [1:4]
                for k = [3:10]

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

                    e = ones(total,1);
                    f_plus = computeRBFKernel(M,M,sigma)*lambda_plus + e*b_plus;
                    f_minus = computeRBFKernel(M,M,sigma)*lambda_minus + e*b_minus;

                    %Find minimum of the distance to two hyperplanes and then classify to
                    %positive or negative. Find accuracy

                    predicted = 2*(min(abs(f_plus),abs(f_minus)) == abs(f_plus))-1;
                    correct = (predicted ==  z);
                    accuracy =  sum(correct)/size(correct,1);
                    
                    fprintf('Accuracy: %f\n\n',accuracy);
                    
                    if(accuracy>max_accuracy)
                       max_accuracy = accuracy;
                       best_c_1 = c_1;
                       best_c_2 = c_2;
                       best_c_3 = c_3;
                       best_sigma = sigma;
                       best_predicted = predicted;
                       best_k = k;
                       best_lambda_plus = lambda_plus;
                       best_b_plus = b_plus;
                       best_lambda_minus = lambda_minus;
                       best_b_minus = b_minus;
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

% Test
test_data =[test_positive;test_negative];

e_1 = ones(size(test_data,1),1);
f_plus = computeRBFKernel(test_data,M,best_sigma)*best_lambda_plus + e_1*best_b_plus;
f_minus = computeRBFKernel(test_data,M,best_sigma)*best_lambda_minus + e_1*best_b_minus;

%Find minimum of the distance to two hyperplanes and then classify to
%positive or negative. Find accuracy

predicted = 2*(min(abs(f_plus),abs(f_minus)) == abs(f_plus))-1;
z1 = [ones(size(test_positive,1),1);ones(size(test_negative,1),1)*-1];
correct = (predicted == z1);
accuracy =  sum(correct)/size(correct,1);
                    
fprintf('Test Accuracy: %f\n\n',accuracy);




