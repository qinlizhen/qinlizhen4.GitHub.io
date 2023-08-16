%% Load the data and preprocess the data-------------
% 将所有测得的数据切分为7500个数据和对数据进行标准化，消除变量之间的量纲关系，使数据具有可比性。
load testdata.mat %其中第一列为时间，第二列至第六列为波高数据，第七列为波浪力数据。
dt=0.01         %这里的每一步的间隔为0.01秒，可根据实际实验采样频率情况来定。
time_lag=500;   %这里指的是根据波高力数据预测5秒后的波浪力，这个间距可以自己根据实际来调整。
N_total=7500;  %这里指的是7500个时间步
t=testdata(:,1);      %时间，共有7500个。
std_scale=std(testdata(:,2:7));  %对波高数据和波浪力数据求标准差
testdata_norm(:,2:7) =normalize(testdata(:,2:7));  %将数据归一化
lag=500;
 %-Step2: Construct a time series with alternating 5-second intervals
 % as the input and output of the neural network.-------
 %% 首先先将时间步分割好
t_input=t(1:end-lag);    %7000=7500-500。1:7000个
t_output=t(lag+1:end);   %501:7500个
 
%% 分割波高数据和波浪力数据
height_input(1:5,:)=testdata_norm(1:end-lag,2:6)';   %因此这里height_input中的每一行是一个波高仪的数据，一共有5个波高仪，每个波高仪收集了1-7500个时间步的波高信息。然后这里波高作为输入，是只截取第1-7000个时间步的数据作为输入。
height_output(1,:)=testdata_norm(lag+1:end,7)'; %这里指的是第501~7500个时间步的波浪力作为输出。
 
%% 决定网络的输入、输出、总数
net_input=height_input;    %1:7000，指代的是波高信息
net_output=height_output;    %501:7500，指代的是波浪力信息
sample_size=length(height_output);    %样本总数7000
% Step3: Neural network training--------------
%% 训练神经网络参数设定
numHiddenUnits =5;    %指定LSTM层的隐含单元个数为5
train_ratio= 0.8;     %划分用于神经网络训练的数据比例为总数的80%
LearnRateDropPeriod=50;  %乘法之间的纪元数由“ LearnRateDropPeriod”控制
LearnRateDropFactor=0.5;  %乘法因子由参“ LearnRateDropFactor”控制，
 
 
%% 定义训练时的时间步。
numTimeStepsTrain = floor(train_ratio*numel(net_input(1,:)));   %一共为7000个*0.8=5600个
 
%% 交替一个时间步，可以交替多个时间步，但这里交替一个时间步的效果其实是最好的，详见那篇开头第二篇建立在python之上的文章。（真正想要改变往后预测时间的长短为lag这个变量。）
XTrain = net_input(:, 1: numTimeStepsTrain+1);  %1~5601，XTrain---input,一共为5601个
YTrain = net_output(:, 2: numTimeStepsTrain+2);   %2~5602，YTrain---expected output，为5601个
 
%% 输入有5个特征，输出有1个特征。
numFeatures =  numel(net_input(:,1));    %5  
numResponses =  numel(net_output(:,1));   %1  
 
layers = [ ...
    sequenceInputLayer(numFeatures)   %输入层为5
    lstmLayer(numHiddenUnits)         %lstm层，构建5层的LSTM模型，
    fullyConnectedLayer(numResponses) %为全连接层，是输出的维数。
    regressionLayer];         %其计算回归问题的半均方误差模块 。即说明这不是在进行分类问题。
 
    options = trainingOptions('adam', ... %指定训练选项，求解器设置为adam， 1000轮训练。
        'MaxEpochs',150, ...    %最大训练周期为150
        'GradientThreshold',1, ...   %梯度阈值设置为 1
        'InitialLearnRate',0.01, ...  %指定初始学习率 0.01
        'LearnRateSchedule','piecewise', ...  %每当经过一定数量的时期时，学习率就会乘以一个系数。
        'LearnRateDropPeriod', LearnRateDropPeriod, ...  
        'LearnRateDropFactor',LearnRateDropFactor, ...  %在50轮训练后通过乘以因子 0.5 来降低学习率。
        'Verbose',0, ...   %如果将其设置为true，则有关训练进度的信息将被打印到命令窗口中,0即是不打印 。
        'Plots','training-progress');   %构建曲线图 ，不想构造就将'training-progress'替换为none
 
 
net = trainNetwork(XTrain,YTrain,layers,options);    %训练神经网络
save('LSTM_net', 'net');            %将net保存为LSTM_net
 
end

 %Step4: Neural network prediction + precision reflection------

 %% 定义Xtest和Ytest，其目的是仿照神经网络训练中XTrain和YTrain的样式去构造一个测试集
%% input_Test指的是实验中得到的波高数据。
%% output_Test指的就是参与跟预测值进行对比的测量值。即expected output
numTimeStepsTrain2 = floor(0.1*sample_size);    %一共为7000个*0.1=700个。可以自己再随意取一个比例，去尝试一下。当然也可以跟上面的numTimeStepsTrain保持一致。
input_Test = net_input(:, numTimeStepsTrain2+1: end-1);   %7000个中取701个~6999，一共是6299个
output_Test = net_output(:, numTimeStepsTrain2+2: end);   %7000个取702:7000，为6299个，因为 numTimeStepsTrain是700.
 
 
%% 这里首先用input_Train来初始化神经网络。经过测试，不是将整个input_Train放进去最好，放其中一点比例即可。
input_Train = net_input(:, floor(numTimeStepsTrain*0.9): numTimeStepsTrain);   %630~700
net = predictAndUpdateState(net, input_Train); 
 
 
%% 预测量预定义
rec_step=numel(output_Test);    %滚动预测6299个。跟后面的j循环有关。
YPred=zeros(rec_step,1);  %6299个预测，1个特征。这个预测是和Test来比对，看是否正确的。
 
 
%% 神经网络预测(这个也是之后实际预测需要用到的)
    for j=1: rec_step
        [net,YPred0] = predictAndUpdateState(net, input_Test(:, j));    %用input_Test代入刚刚用input_Train来更新的网络得到第一个输出并得到对应的预测值。
        YPred(j, :) = YPred0;    %记录输出的预测值。
    end
 
 
%% 神经网络精度反映。这里的Y都是波浪力
YTest = output_Test(:,1:rec_step)';     %这一步可以简化。其实只是一个转置。
NRMSE=sqrt(sum((YPred-YTest).^2)/rec_step/(max(YTest)-min(YTest)));
%Step5：神经网络实际应用-------------
rec_step=2;    %这里只预测两步
input_Test=[1;2;3;4;5,6;4;8;9;7];   %假设第一个时间步波高数据分别为1,2,3,4,5，第二个时间步波高数据分别为6;4;8;9;7。
 
%% 神经网络预测
    for j=1: rec_step
        [net,YPred0] = predictAndUpdateState(net, input_Test(:, j));    %用input_Test代入刚刚用input_Train来更新的网络得到第一个输出并得到对应的预测值。
        YPred(j, :) = YPred0;    %记录输出2个的预测值。
    end
 










