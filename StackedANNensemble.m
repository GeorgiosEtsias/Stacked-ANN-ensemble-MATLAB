clear 
clc
load vinyl_dataset.mat

%% Dividing the vinyl_dataset into 4 input and target sub-sets
% ensemble
datapoints=size(vinylTargets);
datapoints=datapoints(2);
dataset1=round(datapoints/3);
dataset2=2*dataset1;
Input1=vinylInputs(:,1:dataset1); % Used for metalearner training
Target1=vinylTargets(1:dataset1);
Input2=vinylInputs(:,dataset1+1:dataset2); % Used for Level-0 learner training
Target2=vinylTargets(dataset1+1:dataset2);
Input3=vinylInputs(:,dataset2+1:datapoints); % Used for metalearner testing
Target3=vinylTargets(dataset2+1:datapoints);

%% Training Level-0 learners in the subset1 and subset2
nnetworks=4; %numbers of learners per subset
for i=1:nnetworks
net1 = feedforwardnet(randi([1 10],[1 3]));
net2 = train(net1,Input2,Target2,'useParallel','yes','showResources','yes');
Learner0(i).ANN=net2;
end
%% Deriving training data for Level-1 meta-learner
for i=1:nnetworks
    net=Learner0(i).ANN;
    metadata(i,:)=net(Input1);
end
% The metalearner is trained in both the metadata and the Input3 data set
metatraining=[metadata;Input1];
%% Training Level-1 mata-learner
net1 = feedforwardnet([5 5 5]);
net2 = train(net1,metatraining,Target1,'useParallel','yes','showResources','yes');
metalearner=net2;
%% Predicting subset3 with the metalerer and the ANNS
for i=1:nnetworks
    net=Learner0(i).ANN;
    metadata2(i,:)=net(Input3);
    perfNN(i)=perform(net, Target3, net(Input3));
end
metainputs=[metadata2;Input3];
outputs=metalearner(metainputs);
targets=Target3;
perfLevel1 = perform(metalearner, targets, outputs);

