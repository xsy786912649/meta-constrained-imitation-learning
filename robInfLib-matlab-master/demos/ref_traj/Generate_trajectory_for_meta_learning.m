clear; close all;
s=pwd
cd('..');
myColors;
addpath('../fcts/');

%% Extract position and velocity from demos
load('../2Dletters/Z.mat')
demoNum=5;                    % number of demos
demo_dt=0.01;                 % time interval of data
demoLen=size(demos{1}.pos,2); % size of each demo;
demo_dura=demoLen*demo_dt;    % time length of each demo
dim=2;                        % dimension of demo

totalNum=0; 
for i=1:demoNum 
    for j=1:demoLen 
        totalNum=totalNum+1; 
        Data(1,totalNum)=j*demo_dt; 
        Data(2:dim+1,totalNum)=demos{i}.pos(1:dim,j); 
    end
    lowIndex=(i-1)*demoLen+1; 
    upIndex=i*demoLen; 
    for k=1:dim 
        Data(dim+1+k,lowIndex:upIndex)=gradient(Data(1+k,lowIndex:upIndex))/demo_dt; 
    end 
end

%% Extract the reference trajectory
model.nbStates = 12;   % Number of states in the GMM 
model.nbVar =1+2*dim; % Number of variables [t,x1,x2,.. vx1,vx2...] 
model.dt = 0.005;     % Time step duration 
nbData = demo_dura/model.dt; % Length of each trajectory 

model = init_GMM_timeBased(Data, model);
model = EM_GMM(Data, model);
[DataOut, SigmaOut] = GMR(model, [1:nbData]*model.dt, 1, 2:model.nbVar); %see Eq. (17)-(19)

for i=1:nbData
    refTraj(i).t=i*model.dt;
    refTraj(i).mu=DataOut(:,i);
    refTraj(i).sigma=SigmaOut(:,:,i);
end

cd(s);
filename='Z_reftraj.mat';
save(filename,'refTraj')

