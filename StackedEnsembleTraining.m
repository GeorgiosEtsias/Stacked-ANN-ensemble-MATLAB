%----G Etsias November 19-2018----------%
%------Training Stacking Ensemble---------------------------------------%
%The predictions of the ANN ensemble serve as training data to serve
%another ANN (level-1 ANN).
clear
clc
load('EnsembleTainOutp')%predictions of the 20 ANN for training data
load('trainDATAA')%the 87.5% of training data, its last column serves as goal
datasize=size(trainDATAA); % datasize(1) is the dimension we're interested in
StackingTrainingData=zeros(datasize(1),20);%giving zeros to the initial Data matrix
for i=1:20
        intermMatrix=EnsembleTainOutp(i).outp;
        intermMatrix=intermMatrix';
        StackingTrainingData(:,i)=intermMatrix;%It is gonna be the training data of the level-1 ANN
end
%% Creating the level-1 ANN using meta-data of the level-0 ANNs (20 0of them)
trainn=[StackingTrainingData,trainDATAA(:,3)];
goall=trainDATAA(:,5);%the same as in level-0
trainn=trainn';
goall=goall';
net1 = feedforwardnet([10 10 10]);
net1.trainParam.time= 900;
[net2,tr] = train(net1,trainn,goall,'useParallel','yes','showResources','no');
        
%% Calculating performance for the whole data set,in the final epoch 
inputs=trainn;
targets=goall;
outputs = net2(inputs);
perf = perform(net2, targets, outputs)


