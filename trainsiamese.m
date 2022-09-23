clear;
all_data = readmatrix("sequence.csv");
data_1=readmatrix('s1.txt','OutputType','uint8'); %some extra data due to space in total 1560 data per sequence reshaped to [40 39]
data_2=readmatrix('s2.txt','OutputType','uint8');
data_1=data_1(:,1:900);
data_2=data_2(:,1:900);
target_1=double(all_data(:,3));
target_2=double(all_data(:,4));
X_train1=reshape(data_1,30,30,[]);
Y_train1=target_1;
X_train2=reshape(data_2,30,30,[]);
Y_train2=target_2;
sum(data_1(1,:))% should be equal to count of sequenece length 16-20
%%
layers = [
    imageInputLayer([30 30],Normalization="none")
    convolution2dLayer(3,64,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(3,128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(5,128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    fullyConnectedLayer(1024,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")];

lgraph = layerGraph(layers);
% To train the network with a custom training loop and enable automatic differentiation, convert the layer graph to a dlnetwork object.
net = dlnetwork(lgraph);
% Create the weights for the final fullyconnect operation. Initialize the weights by sampling a random selection from a narrow normal distribution with standard deviation of 0.01.
fcWeights = dlarray(0.01*randn(1,1024));
fcBias = dlarray(0.01*randn(1,1));

fcParams = struct(...
    "FcWeights",fcWeights,...
    "FcBias",fcBias);
%%
numIterations = 2000;
learningRate = 6e-4;
gradDecay = 0.9;
gradDecaySq = 0.99;
executionEnvironment = "auto";
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
trailingAvgSubnet = [];
trailingAvgSqSubnet = [];
trailingAvgParams = [];
trailingAvgSqParams = [];
start = tic;
batchSize=8;
% Loop over mini-batches.
for iteration = 1:numIterations

    % Extract mini-batch of image pairs and pair labels
    [X1,X2,pairLabels] = getSiameseBatch(X_train1,Y_train1,X_train2,Y_train2,batchSize);

    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % "SSCB" (spatial, spatial, channel, batch) for image data
    X1 = dlarray(X1,"SSCB");
    X2 = dlarray(X2,"SSCB");

    % If training on a GPU, then convert data to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        X1 = gpuArray(X1);
        X2 = gpuArray(X2);
%         fprintf('%f',sum(sum(X1)));
%         fprintf('%f',sum(sum(X2)));
    end

    % Evaluate the model loss and gradients using dlfeval and the modelLoss
    % function listed at the end of the example.
    [loss,gradientsSubnet,gradientsParams] = dlfeval(@modelLoss,net,fcParams,X1,X2,pairLabels);

    % Update the Siamese subnetwork parameters.
    [net,trailingAvgSubnet,trailingAvgSqSubnet] = adamupdate(net,gradientsSubnet, ...
        trailingAvgSubnet,trailingAvgSqSubnet,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the fullyconnect parameters.
    [fcParams,trailingAvgParams,trailingAvgSqParams] = adamupdate(fcParams,gradientsParams, ...
        trailingAvgParams,trailingAvgSqParams,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the training loss progress plot.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    lossValue = double(loss);
    addpoints(lineLossTrain,iteration,lossValue);
    title("Elapsed: " + string(D))
    drawnow
end
%%
accuracy = zeros(1,100);
accuracyBatchSize = 10;

for i = 1:10
    % Extract mini-batch of image pairs and pair labels
    [X1,X2,pairLabelsAcc] = getSiameseBatch(X_train1,Y_train1,X_train2,Y_train2,accuracyBatchSize); %later replace with X_test1. with X_train must show ~100% training accuracy.

    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % "SSCB" (spatial, spatial, channel, batch) for image data.
    X1 = dlarray(X1,"SSCB");
    X2 = dlarray(X2,"SSCB");

    % If using a GPU, then convert data to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        X1 = gpuArray(X1);
        X2 = gpuArray(X2);
    end

    % Evaluate predictions using trained network
    Y = predictSiamese(net,fcParams,X1,X2); %final calculated Abundance Ratio
%     Y = round(Y);
%     pairLabelsAcc=round(pairLabelsAcc);

    % Compute average accuracy for the minibatch
    accuracy(i) = sum(Y == pairLabelsAcc)/accuracyBatchSize;
    fprintf('True %f',pairLabelsAcc)
    fprintf('Predicted %f',Y)
end
%%

function Y = forwardSiamese(net,fcParams,X1,X2)
% forwardSiamese accepts the network and pair of training images, and
% returns a prediction of the probability of the pair being similar (closer
% to 1) or dissimilar (closer to 0). Use forwardSiamese during training.

% Pass the first image through the twin subnetwork
Y1 = forward(net,X1);
Y1 = sigmoid(Y1);

% Pass the second image through the twin subnetwork
Y2 = forward(net,X2);
Y2 = sigmoid(Y2);

% Y = abs(Y1-Y2);% Subtract the feature vectors
Y = (log(Y1)-log(Y2)); % not abs value THIS IS THE MOST IMPORTANT CHANGE INSTEAD OF SUBTRACTION, LOG value is provided i.e loga-logb i.e. log(a/b) is done.

% Pass the result through a fullyconnect operation
Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);

% Y = sigmoid(Y); % GENERALLY Convert to probability between 0 and 1, but we wanted the real score of Y so not converted later in cross entropy this 'loss' i.e Y value is calculated.

end
%The function predictSiamese uses the trained network to make predictions about the similarity of two images. The function is similar to the function forwardSiamese, defined previously. However, predictSiamese uses the predict function with the network instead of the forward function, because some deep learning layers behave differently during training and prediction. Within this example, the function predictSiamese is introduced in the section Evaluate the Accuracy of the Network.
function Y = predictSiamese(net,fcParams,X1,X2)
% predictSiamese accepts the network and pair of images, and returns a
% prediction of the probability of the pair being similar (closer to 1) or
% dissimilar (closer to 0). Use predictSiamese during prediction.

% Pass the first image through the twin subnetwork.
Y1 = predict(net,X1);
Y1 = sigmoid(Y1);

% Pass the second image through the twin subnetwork.
Y2 = predict(net,X2);
Y2 = sigmoid(Y2);

% Y = abs(Y1-Y2);% Subtract the feature vectors% Subtract the feature vectors.
Y = (log(Y1)-log(Y2)); % THIS IS THE MOST IMPORTANT CHANGE INSTEAD OF SUBTRACTION, LOG value is provided i.e loga-logb i.e. log(a/b) is done.

% Pass result through a fullyconnect operation.
Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);

% Convert to probability between 0 and 1.
% Y = sigmoid(Y);

end
% Model Loss Function
% The function modelLoss takes the Siamese dlnetwork object net, a pair of mini-batch input data X1 and X2, and the label indicating whether they are similar or dissimilar. The function returns the binary cross-entropy loss between the prediction and the ground truth and the gradients of the loss with respect to the learnable parameters in the network. Within this example, the function modelLoss is introduced in the section Define Model Loss Function.
function [loss,gradientsSubnet,gradientsParams] = modelLoss(net,fcParams,X1,X2,pairLabels)

% Pass the image pair through the network.
Y = forwardSiamese(net,fcParams,X1,X2);

% Calculate binary cross-entropy loss.
loss = binarycrossentropy(Y,pairLabels);
[gradientsSubnet,gradientsParams] = dlgradient(loss,net.Learnables,fcParams);

end
%loss function
function loss = binarycrossentropy(Y,pairLabels)

% Get precision of prediction to prevent errors due to floating point
precision = underlyingType(Y);
% Convert values less than floating point precision to eps.
Y(Y < eps(precision)) = eps(precision);
pairLabels(pairLabels < eps(precision)) = eps(precision);
% 
% % Convert values between 1-eps and 1 to 1-eps.
Y(Y > 1 - eps(precision)) = 1 - eps(precision);

pairLabels(pairLabels > 1 - eps(precision)) = 1 - eps(precision);
% Calculate binary cross-entropy loss for each pair
% loss = -pairLabels.*log(Y) - (1 - pairLabels).*log((1 - Y));
loss = power((pairLabels-Y),2);
% Sum over all pairs in minibatch and normalize.
loss = sum(loss)/numel(pairLabels);

end

function [X1,X2,pairScore] = getSiameseBatch(X11,Y1,X22,Y2,miniBatchSize)

pairLabels1 = zeros(1,miniBatchSize);
pairLabels2 = zeros(1,miniBatchSize);
imgSize = size(X11(:,:,1));
X1 = zeros([imgSize 1 miniBatchSize],"single");
X2 = zeros([imgSize 1 miniBatchSize],"single");
% X1=double(X11);
% X2=double(X22);
        for i = 1:miniBatchSize
%     choice = rand(1);
%     if choice < 0.5
        [pairIdx1,pairLabels1(i)] = getfirstIntensity(Y1);
        X1(:,:,i) = X11(:,:,pairIdx1);
%     else
        [pairIdx2,pairLabels2(i)] = getsecondIntensity(Y2);
        X2(:,:,i) = X22(:,:,pairIdx2);
%     end
        pairScore(i)=(log(pairLabels1(i))-log(pairLabels2(i))); %CALCULATED ABUNDANCE RATIO
         
         end

end

function [pairIdx1,pairLabel] = getfirstIntensity(classLabel)

% Find all unique classes.
classes = (classLabel);

% Choose a class randomly which will be used to get a similar pair.
classChoice = randi(numel(classes));

% Find the indices of all the observations from the chosen class.
idxs = find(classLabel==classes(classChoice));

% Randomly choose two different images from the chosen class.
pairIdxChoice = randperm(numel(idxs),1);
pairIdx1 = idxs(pairIdxChoice(1));
% pairIdx2 = idxs(pairIdxChoice(1));
pairLabel = classes(pairIdx1);

end

function  [pairIdx2,label] = getsecondIntensity(classLabel)

% Find all unique classes.
classes = (classLabel);

% Choose two different classes randomly which will be used to get a
% dissimilar pair.
classesChoice = randperm(numel(classes),1);

% Find the indices of all the observations from the first and second
% classes.
% idxs1 = find(classLabel==classes(classesChoice(1)));
idxs2 = find(classLabel==classes(classesChoice(1)));

% Randomly choose one image from each class.
% pairIdx1Choice = randi(numel(idxs1));
pairIdx2Choice = randi(numel(idxs2));
% pairIdx1 = idxs1(pairIdx1Choice);
pairIdx2 = idxs2(pairIdx2Choice);
label = classes(pairIdx2);
end
%%