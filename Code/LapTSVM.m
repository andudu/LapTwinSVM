load('2moons.mat');
M =x;
total = size(M,1);
%1:positive
%-1: negative
%First manually make entries in dataset 0- unlabelled

positive_indices=find(y==1);
A= M(positive_indices,:);

negative_indices=find(y==-1);
B= M(negative_indices,:);

% Calculate L using D,W (First we need to find W)

c_1 = 4;
c_2 = 0.024;
%c_3 = 10;

lamda_plus = zeros(total,1);
b_plus = zeros(total,1);


lamda_minus = zeros(size,1);
b_minus = zeros(size,1);


[lamda_plus ; b_plus] = positiveLapSVM(L,A,B, M,c_1,c_2,c_3);
[lamda_minus ; b_minus] = negativeLapSVM(L,A,B, M,c_1,c_2,c_3);

%distance_x = abs(K())

%Find minimum of the distance to two hyperplanes and then classify to
%positive or negative. Find accuracy

