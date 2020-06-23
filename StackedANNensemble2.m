clear 
clc
load vinyl_dataset.mat % available in Deep Learning Toolbox™ 

%% Rearranging the vinyl_dataset randomly into 3 subset
data=[vinylInputs;vinylTargets];
datapoints=size(data);
randomdata=zeros(datapoints(1),datapoints(2));
datapoints=datapoints(2);
shuffle = randperm(datapoints);
% Create a randomly ordered vector of indices
for i=1:datapoints
    randomdata(:,i)=data(:,shuffle(i));
end
dataset1=round(datapoints/3);
dataset2=2*dataset1;
subset1=randomdata(:,1:dataset1); % Used for Level-0 learner training
subset2=randomdata(:,dataset1+1:dataset2); % Used for metalearner training 
subset3=randomdata(:,dataset2+1:datapoints); % Used for metalearner testing
% dividing inputs and targets
Input1=subset1(1:16,:);
Target1=subset1(17,:);
Input2=subset2(1:16,:);
Target2=subset2(17,:);
Input3=subset3(1:16,:);
Target3=subset3(17,:);


%% Training Level-0 learners in the subset1 and subset2
nnetworks=20; %numbers of learners per subset
for i=1:nnetworks
net1 = feedforwardnet(randi([1 10],[1 3]));
net2 = train(net1,Input1,Target1,'useParallel','yes','showResources','no');
Learner0(i).ANN=net2;
end
%% Deriving training data for Level-1 meta-learner
for i=1:nnetworks
    net=Learner0(i).ANN;
    metadata(i,:)=net(Input2);
end
% The metalearner is trained in both the metadata and the Input3 data set
metatraining=[metadata;Input2];
%% Training Level-1 mata-learner
net1 = feedforwardnet([5 5 5]);
net2 = train(net1,metatraining,Target2,'useParallel','yes','showResources','no');
metalearner=net2;
%% Predicting subset3 with the metalerer and the ANNS
for i=1:nnetworks
    net=Learner0(i).ANN;
    metadata2(i,:)=net(Input3);
    perfLevel0(i)=perform(net, Target3, net(Input3));
end
metainputs=[metadata2;Input3];
outputs=metalearner(metainputs);
targets=Target3;
perfLevel1 = perform(metalearner, targets, outputs);

%% Comparing metalearners performance to the one of the most succesfull ANN
perfratio=perfLevel1/(min(perfLevel0))*100;