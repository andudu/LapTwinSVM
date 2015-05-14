
% Graph based Label Propagation- Based on "Semi-Supervised Learning Using Gaussian Fields 
% and Harmonic Functions".  Xiaojin Zhu, Zoubin Ghahramani, John Lafferty.  
% The Twentieth International Conference on Machine Learning (ICML-2003).
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
 
best_k = -1;
best_sigma = -1;
best_accuracy = -1;
% Vary k,sigma on some range.
for k= [1:20]
   for sigma = [0.25:0.25:4]
 
   %k= 4;
   % sigma = 1;

       total =size(M,1);

       IDX = knnsearch(M,M,'K',k);
        W = zeros(total,total);
        for i = 1:total
           for j = 1:k
               val = exp(-norm(M(i,:)-M(IDX(i,j),:))/(2*sigma^2));
               W(i,IDX(i,j)) = val;
               W(IDX(i,j),i) = val;
           end
        end


       L = size(labelled_positive,1) + size(labelled_negative,1);                 
       label = zeros(size(L),2);

       for i =1 : L
           if(i<=size(labelled_positive,1))
               label(i,:) =[1 0]';
           end
           if(i>  size(labelled_positive,1))
               label(i,:)=[0 1]';
           end
       end
          [fu, fu_CMN] = harmonic_function(W, label);
          full_labels = [label; fu];   
          
          
        alllabels = zeros(size(full_labels,1),1);
        for i=1:size(alllabels,1)
            [val idx] = max(full_labels(i,:));
            if(idx==2)
            alllabels(i) = -1;
            end
             if(idx==1)
               alllabels(i) = 1;
             end
        end
            
          
       

       test = [test_positive; test_negative];

   
    
       IDX = knnsearch(test,M,'K',k);
        W = zeros(size(test,1),total);
        for i = 1:size(test,1)
           for j = 1:k
               val = exp(-norm(M(i,:)-M(IDX(i,j),:))/(2*sigma^2));
               W(i,IDX(i,j)) = val;
               W(IDX(i,j),i) = val;
           end
        end
       
        
        correct =0;
        
        for i=1:size(test,1)
            f_x=0;
            w_sum =0;
            for j=1:k
                f_x= f_x + W(i,j)*alllabels(IDX(i,j));
                w_sum = w_sum + W(i,j);
            end
              f = f_x/w_sum;
               if and (i<=size(test_positive,1), f>=0)
                   correct = correct+1;
               end
               if(and(i>size(test_positive,1),f<=0))
                   correct = correct +1;
               end
        end
        accuracy = correct/ size(test,1);        
       
       
       fprintf('Test Accuracy: %f, %f, %f\n' , accuracy, k, sigma);

       if accuracy>best_accuracy
           best_k = k;
           best_sigma = sigma;
           best_accuracy = accuracy;
       end
       
    end
end

fprintf('Best test accuracy: %f, best_k: %f, best_sigma: %f\n' , best_accuracy, best_k, best_sigma);
      
      
                    
                    
                

