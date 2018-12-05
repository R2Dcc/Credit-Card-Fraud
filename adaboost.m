%{
Copyright (c) 2017, Bhartendu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution
* Neither the name of Dept. of Space nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

Refer to: https://www.iist.ac.in/sites/default/files/people/in12167/adaboost.pdf
%}
%{
Modificado por: Cristian Cofré S.
04/12/2018
%}
function [ada_train, ada_test]= adaboost(Xtrain,Ytrain, Xtest)
%[ada_train1, ada_test1]= adaboost(train_1,data_train1(:,31), data_test(:,[10 11 12 14 16 17]));
%train_1 = data_train1(:,[10 11 12 14 16 17]);---------->1432X6
%data_train1(:,31)-------------------------------------->1432X1
%data_test(:,[10 11 12 14 16 17])----------------------->113923x6
% AdaBoost function 
% (X_train-> input: training set)
% (Y_train-> target)
% (Xtest-> input: testing set)
% (ada_train-> label: training set)
% (ada_test-> label: testing set)

% Choosen Weak classifiers:
% 1. GDA
% 2. knn (NumNeighbors = 30)
% 3. Naive Bayes
% 4. Logistic Regression
% 5. SVM (rbf)

N=size(Xtrain,1); %1432
a=[Xtrain Ytrain]; %Dataset train matrix = 1432x7

D=(1/N)*ones(N,1);
Dt=[]; h_=[];

Classifiers=10; %N° de clasificadores
eps=zeros(Classifiers,1);

for T=1:Classifiers
    p_min=min(D);
    p_max=max(D);
    
    for i=1:length(D)
        p = (p_max-p_min)*rand(1) + p_min;
        
        if D(i)>=p
            d(i,:)=a(i,:);
        end
        
        t=randi(size(d,1));
        Dt=[Dt ;d(t,:)];
    end

    X=Dt(:,1:end-1);%---->train_1 DATA/PREDICTOR
    Y=Dt(:,end);%-------->data_train1(:,31) CLASES
        
    if T==1
		%SVM
		svm1_in = fitcsvm(X,Y,'Solver','SMO');
		svm1_out=predict(svm1_in,X);
		h=svm1_out;
        Dt=Dt(length(Dt)+1:end,:);
        % knn with (30 Nearest Neighbour)
        %knn_in=fitcknn(X,Y,'NumNeighbors',30);
        %knn_out=predict(knn_in, X);
        %h=knn_out;
        %Dt=Dt(length(Dt)+1:end,:);
    end
    
    if T==2
		%SVM
		svm2_in = fitcsvm(X,Y,'Solver','SMO');
		svm2_out=predict(svm2_in,X);
		h=svm2_out;
        Dt=Dt(length(Dt)+1:end,:);
        % nb
        %nb_in=fitcnb(X,Y);
        %nb_out=predict(nb_in, X);
        %h=nb_out;
        %Dt=Dt(length(Dt)+1:end,:);
    end
    
    if T==3
		%SVM
		svm3_in = fitcsvm(X,Y,'Solver','SMO');
		svm3_out=predict(svm3_in,X);
		h=svm3_out;
        Dt=Dt(length(Dt)+1:end,:);
        % logistic regression
        %linear_in=fitclinear(X,Y,'Learner','logistic');
        %linear_out=predict(linear_in, X);
        %h=linear_out;
        %Dt=Dt(length(Dt)+1:end,:);
    end
	
	if T==4
		%SVM
		svm4_in = fitcsvm(X,Y,'Solver','SMO');
		svm4_out=predict(svm4_in,X);
		h=svm4_out;
		Dt=Dt(length(Dt)+1:end,:);
		% logistic regression
		%linear_in=fitclinear(X,Y,'Learner','logistic');
		%linear_out=predict(linear_in, X);
		%h=linear_out;
		%Dt=Dt(length(Dt)+1:end,:);
    end    
	
	if T==5
		%SVM
		svm5_in = fitcsvm(X,Y,'Solver','SMO');
		svm5_out=predict(svm5_in,X);
		h=svm5_out;
        Dt=Dt(length(Dt)+1:end,:);
        % logistic regression
        %linear_in=fitclinear(X,Y,'Learner','logistic');
        %linear_out=predict(linear_in, X);
        %h=linear_out;
        %Dt=Dt(length(Dt)+1:end,:);
    end
    
    if T==6
		% tree
		tree1_in=fitctree(X,Y);
		tree1_out= predict(tree1_in,X);
		h=tree1_out;
		Dt=Dt(length(Dt)+1:end,:);
        % svm 'rbf'
        %svm_in=fitcsvm(X,Y,'KernelFunction','rbf');
        %svm_out=predict(svm_in, X);
        %h=svm_out;
        %Dt=Dt(length(Dt)+1:end,:);
    end  
    
	if T==7
        % tree
		tree2_in=fitctree(X,Y);
		tree2_out= predict(tree2_in,X);
		h=tree2_out;
		Dt=Dt(length(Dt)+1:end,:);
		% gda
        %gda_in=fitcdiscr(X,Y);
        %gda_out=predict(gda_in, X);
        %h=gda_out;
        %Dt=Dt(length(Dt)+1:end,:);
    end
	if T==8
        % tree
		tree3_in=fitctree(X,Y);
		tree3_out= predict(tree3_in,X);
		h=tree3_out;
		Dt=Dt(length(Dt)+1:end,:);
		% gda
        %gda_in=fitcdiscr(X,Y);
        %gda_out=predict(gda_in, X);
        %h=gda_out;
        %Dt=Dt(length(Dt)+1:end,:);
    end
	
	if T==9
        % tree
		tree4_in=fitctree(X,Y);
		tree4_out= predict(tree4_in,X);
		h=tree4_out;
		Dt=Dt(length(Dt)+1:end,:);
		% gda
        %gda_in=fitcdiscr(X,Y);
        %gda_out=predict(gda_in, X);
        %h=gda_out;
        %Dt=Dt(length(Dt)+1:end,:);
    end
	
	if T==10
        % tree
		tree5_in=fitctree(X,Y);
		tree5_out= predict(tree5_in,X);
		h=tree5_out;
		Dt=Dt(length(Dt)+1:end,:);
		% gda
        %gda_in=fitcdiscr(X,Y);
        %gda_out=predict(gda_in, X);
        %h=gda_out;
        %Dt=Dt(length(Dt)+1:end,:);
    end
	
    h_=[h_ h];

    % weighted error
    for i=1:length(Y)
        if (h_(i,T)~=Y(i))
            eps(T)=eps(T)+D(i,:); 
        end  
    end
    
    % Hypothesis weight
    alpha(T)=0.5*log((1-eps(T))/eps(T));
    
    % Update weights
    D=D.*exp((-1).*Y.*alpha(T).*h);
    D=D./sum(D);
end

% final vote
%{
H(:,1)=predict(gda_in, Xtrain);
H(:,2)=predict(knn_in, Xtrain);
H(:,3)=predict(nb_in, Xtrain);
H(:,4)=predict(linear_in, Xtrain);
H(:,5)=predict(svm_in, Xtrain);
ada_train(:,1)=sign(H*alpha');
%}
H(:,1)=predict(svm1_in, Xtrain);
H(:,2)=predict(svm2_in, Xtrain);
H(:,3)=predict(svm3_in, Xtrain);
H(:,4)=predict(svm4_in, Xtrain);
H(:,5)=predict(svm5_in, Xtrain);
H(:,6)=predict(tree1_in, Xtrain);
H(:,7)=predict(tree2_in, Xtrain);
H(:,8)=predict(tree3_in, Xtrain);
H(:,9)=predict(tree4_in, Xtrain);
H(:,10)=predict(tree5_in, Xtrain);
ada_train(:,1)=sign(H*alpha');

% for test set
%{
Htest(:,1)=predict(gda_in, Xtest);
Htest(:,2)=predict(knn_in, Xtest);
Htest(:,3)=predict(nb_in, Xtest);
Htest(:,4)=predict(linear_in, Xtest);
Htest(:,5)=predict(svm_in, Xtest);
ada_test(:,1)=sign(Htest*alpha');
%}
Htest(:,1)=predict(svm1_in, Xtest);
Htest(:,2)=predict(svm2_in, Xtest);
Htest(:,3)=predict(svm3_in, Xtest);
Htest(:,4)=predict(svm4_in, Xtest);
Htest(:,5)=predict(svm5_in, Xtest);
Htest(:,6)=predict(tree1_in, Xtest);
Htest(:,7)=predict(tree2_in, Xtest);
Htest(:,8)=predict(tree3_in, Xtest);
Htest(:,9)=predict(tree4_in, Xtest);
Htest(:,10)=predict(tree5_in, Xtest);
ada_test(:,1)=sign(Htest*alpha');

end

