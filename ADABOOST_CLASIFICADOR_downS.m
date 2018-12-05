%Import datos a workspace como 'creditcard'
%creditcard = readtable('creditcard.csv');
%creditcard = cell2table(T);
%busca en tabla creditcard, columna Class, Class=0
rows0 = creditcard.Class==0;
%busca en tabla creditcard, columna Class, Class=1
rows1 = creditcard.Class==1;
class_0 = creditcard(rows0,:); %solo filas con clase 0 (284315x31)
class_1 = creditcard(rows1,:); %solo filas con clase 1 (492x31)

%Distribuir variabilidad uniformemente de cada clase
clase_0 = random_by_row_matrix(class_0); %clase 0 distruibuida uniformemente de forma aleatoria
clase_1 = random_by_row_matrix(class_1); %clase 1 distruibuida uniformemente de forma aleatoria

%Separar en:
% 60% datos entrenamiento
% 40% datos de test
a_train = clase_0(1:170589,:); %170589x31
b_train = clase_1(1:295,:);		%295x31
a_train = table2array(a_train); %matrix
b_train = table2array(b_train); %matrix
c_test = clase_0(170590:284315,:); %113726x31
d_test = clase_1(296:492,:);	   %197x31

%data_train = [a_train;b_train];
data_test = [c_test;d_test];

%Distribuir datos uniformemente de forma aleatoria
%data_train = random_by_row_matrix(data_train); 
%170884 x 31
data_test = random_by_row_matrix(data_test); %tabla
data_test = table2array(data_test); %matrix
%113923 x 31


n = 150;
rnd = randperm(n-1,5);
dt1 = downsample(a_train,n,rnd(1)); % 1137x31
dt2 = downsample(a_train,n,rnd(2));
dt3 = downsample(a_train,n,rnd(3));
dt4 = downsample(a_train,n,rnd(4));
dt5 = downsample(a_train,n,rnd(5));
data_train11 = [dt1;b_train];%----------------------->1432X31
data_train1 = random_by_row_matrix(data_train11);
data_train22 = [dt2;b_train];
data_train2 = random_by_row_matrix(data_train22);
data_train33 = [dt3;b_train];
data_train3 = random_by_row_matrix(data_train33);
data_train44 = [dt4;b_train];
data_train4 = random_by_row_matrix(data_train44);
data_train55 = [dt5;b_train];
data_train5 = random_by_row_matrix(data_train55);

train_1 = data_train1(:,[10 11 12 14 16 17]); %1432X6
train_2 = data_train2(:,1:30);
train_3 = data_train3(:,[10 11 12 14 16 17]);
train_4 = data_train4(:,1:30);
train_5 = data_train5(:,[10 11 12 14 16 17]);
% AdaBoost function 
% (X_train-> input: training set)----> train_1,train_2, etc
% (Y_train-> target)-----------------> data_train1(:,31) , data_train2(:,31) , etc
% (Xtest-> input: testing set)------->
% (ada_train-> label: training set)-->predicted label
% (ada_test-> label: testing set)---->real label
[ada_train1, ada_test1]= adaboost(train_2,data_train2(:,31), data_test(:,1:30));
figure
cm1 = confusionchart(data_train2(:,31),ada_train1,'RowSummary','row-normalized');
cm1.Title = 'Train 1 with Downsampling, ALL predictors';
figure
cm2 = confusionchart(data_test(:,31),ada_test1,'RowSummary','row-normalized');
cm2.Title = 'Test 1 w/downsampling, ALL predictors';
%%%%%%%%%%%
[ada_train2, ada_test2]= adaboost(train_5,data_train5(:,31), data_test(:,[10 11 12 14 16 17]));
figure
cm1 = confusionchart(data_train5(:,31),ada_train2,'RowSummary','row-normalized');
cm1.Title = 'Train 1 with Downsampling, predictors V10,11,12,14,16,17';
figure
cm2 = confusionchart(data_test(:,31),ada_test2,'RowSummary','row-normalized');
cm2.Title = 'Test 1 w/downsampling, predictors V10,11,12,14,16,17';




%{
labels_s = svm_train(a_train, b_train, data_test);
labels_t = trees_train(a_train, b_train, data_test);
l1 = labels_t(:,1);
l2 = labels_t(:,2);
l3 = labels_t(:,3);
l4 = labels_t(:,4);
l5 = labels_t(:,5);

cm1.Title = 'Tree 1';
cm1.RowSummary = 'row-normalized';
cm1.ColumnSummary = 'column-normalized';
%}


