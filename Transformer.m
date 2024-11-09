%% Inertial DATA FEATURE EXTRACTION

IMDS1 = imageDatastore('FullAugmentedSignalImages\', 'IncludeSubfolders', true, ...
      'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
example_image = readimage(IMDS1, 1);                      % read one example image
numChannels = size(example_image, 3);                     % get color information
numImageCategories = numel(categories(IMDS1.Labels));     % get category labels
[trainingDS1, validationDS1] = splitEachLabel(IMDS1, 0.8, 'randomize'); % generate training and validation set
LabelCnt = countEachLabel(IMDS1);
load('XONet_FullAugmented.mat');
XONet1 = XONet;
layer = 'fc_2';
featuresTrain1 = activations(XONet1, trainingDS1, layer, 'OutputAs', 'rows');
featuresTest1 = activations(XONet1, validationDS1, layer, 'OutputAs', 'rows');
YTrain1 = trainingDS1.Labels;
YTest1 = validationDS1.Labels;

%% Prewitt images feature extraction

IMDS2 = imageDatastore('Prewitt_SignalImages\', 'IncludeSubfolders', true, ...
      'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
numImageCategories = numel(categories(IMDS2.Labels));   % get category labels
[trainingDS2, validationDS2] = splitEachLabel(IMDS2, 0.8, 'randomize'); % generate training and validation set
LabelCnt = countEachLabel(IMDS2);
load('XONet_Prewitt_Signal.mat');
XONet2 = XONet;
layer = 'fc';
featuresTrain2 = activations(XONet2, trainingDS2, layer, 'OutputAs', 'rows');
featuresTest2 = activations(XONet2, validationDS2, layer, 'OutputAs', 'rows');
YTrain2 = trainingDS2.Labels;
YTest2 = validationDS2.Labels;

%% FUSING THE features by Concatenation
FeaturesTrain1 = [featuresTrain1; featuresTrain2];
FeaturesTest1 = [featuresTest1; featuresTest2];
YTrain = [YTrain1; YTrain2];
YTest = [YTest1; YTest2];

%% DEPTH DATA FEATURE EXTRACTION

IMDS3 = imageDatastore('Depth_227x227x3\', 'IncludeSubfolders', true, ...
      'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
example_image = readimage(IMDS3, 1);                      % read one example image
numChannels = size(example_image, 3);                     % get color information
numImageCategories = numel(categories(IMDS3.Labels));     % get category labels
[trainingDS3, validationDS3] = splitEachLabel(IMDS3, 0.8, 'randomize'); % generate training and validation set
LabelCnt = countEachLabel(IMDS3);
load('XONet_depth.mat');
XONet3 = XONet;
layer = 'fc8';
featuresTrain3 = activations(XONet3, trainingDS3, layer, 'OutputAs', 'rows');
featuresTest3 = activations(XONet3, validationDS3, layer, 'OutputAs', 'rows');
YTrain3 = trainingDS3.Labels;
YTest3 = validationDS3.Labels;

%% Prewitt depth feature extraction

IMDS4 = imageDatastore('Depth_Prewitt\', 'IncludeSubfolders', true, ...
      'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
numImageCategories = numel(categories(IMDS4.Labels));   % get category labels
[trainingDS4, validationDS4] = splitEachLabel(IMDS4, 0.8, 'randomize'); % generate training and validation set
LabelCnt = countEachLabel(IMDS4);
load('XONet_Prewitt_depth.mat');
XONet4 = XONet;
layer = 'fc8';
featuresTrain4 = activations(XONet4, trainingDS4, layer, 'OutputAs', 'rows');
featuresTest4 = activations(XONet4, validationDS4, layer, 'OutputAs', 'rows');
YTrain4 = trainingDS4.Labels;
YTest4 = validationDS4.Labels;

%% FUSING THE features by Concatenation
FeaturesTrain2 = [featuresTrain3; featuresTrain4];
FeaturesTest2 = [featuresTest3; featuresTest4];
YYTrain = [YTrain3; YTrain4];
YYTest = [YTest3; YTest4];

%% Define the Transformer Layer
function output = transformerLayer(inputs, n_heads, num_layers)
    size(inputs)
    d_model = size(inputs, 2)  % Dimensionalidad de las características
    n_heads
    Q = randn(d_model, d_model); % Matriz de pesos para Query
    K = randn(d_model, d_model); % Matriz de pesos para Key
    V = randn(d_model, d_model); % Matriz de pesos para Value
    
  
    size(Q)
    size(K)
    size(V)
   
    
    % Calcular las puntuaciones de atención (producto punto entre Q y K)
    attention_scores = inputs * Q' .* inputs * K';
    
    % Aplicar softmax para normalizar las puntuaciones de atención
    attention_probs = softmax(attention_scores, 2); 
    
    % Aplicar atención a los valores
    attention_output = attention_probs * V';  
    
    % Pasar por la red feedforward
    output = feedForwardLayer(attention_output, num_layers);
end

function output = feedForwardLayer(inputs, num_layers)
    % Capa feedforward básica (FC -> ReLU -> FC)
    class(inputs)
    inputs = cast(inputs, 'double');
    class(inputs)
    for i = 1:num_layers
        %inputs = relu(inputs);  % Aplicar ReLU
        inputs = fullyConnectedLayer(inputs, 20);  
    end
    output = inputs;
end

function output = fullyConnectedLayer(inputs, num_neurons)
    % Capa totalmente conectada
    weights = randn(size(inputs, 2), num_neurons);  
    bias = randn(1, num_neurons);  
    output = inputs * weights + bias;
end

function output = softmax(input, dim)
    % Función Softmax para normalizar los scores
    e_x = exp(input - max(input, [], dim));
    output = e_x ./ sum(e_x, dim);
end

%% FTMT Function to Fuse Modalities
function output_features = FTMT(features1, features2, n_heads, num_layers)
    concatenated_features = [features1; features2];
    output_features = transformerLayer(concatenated_features, n_heads, num_layers);
end

%% MCANet Function for Contrastive Alignment
function [aligned_features, loss] = MCANet(features1, features2, labels, margin)
    % MCANet: Multimodal Contrastive Alignment Network
    features1 = normalize(features1, 2);
    features2 = normalize(features2, 2);
    
    size(features1)
    size(features2)
    % Verifica las dimensiones
    [n_samples1, n_features] = size(features1);
    [n_samples2, ~] = size(features2);
    
    % Si features1 tiene menos filas que features2, rellenamos features1
    if n_samples1 < n_samples2
        padding = n_samples2 - n_samples1; % Número de filas a agregar
        features1 = padarray(features1, [padding, 0], 'post'); % Rellenar filas con ceros

        size(features1)
        size(features2)
    
    elseif n_samples2 < n_samples1
        padding = n_samples1 - n_samples2; % Número de filas a agregar
        features2 = padarray(features2, [padding, 0], 'post'); % Rellenar filas con ceros
    end
    cosine_sim = sum(features1 .* features2, 2); % Similitud de coseno
    
    positive_pairs = 1 %(labels == 1);  
    negative_pairs = 0 %(labels == 0); 
    
    loss_pos = sum(positive_pairs .* (1 - cosine_sim).^2);
    loss_neg = sum(negative_pairs .* max(0, cosine_sim - margin).^2);
    
    loss = loss_pos + loss_neg;
    aligned_features = [features1, features2];
end

%% Apply FTMT for the concatenated features
n_heads = 5;  % Number of attention heads
num_layers = 4;  % Number of Transformer layers
FeatureTrainFTMT = FTMT(FeaturesTrain1, FeaturesTrain2, n_heads, num_layers);
FeatureTestFTMT = FTMT(FeaturesTest1, FeaturesTest2, n_heads, num_layers);

%% Apply MCANet for feature alignment
margin = 0.2;  % Contrastive margin
[AlignedTrainFeatures, AlignedTestFeatures] = MCANet(FeaturesTrain1, FeaturesTrain2, YTrain, margin);

%% Concatenate features from FTMT and MCANet
size(FeatureTrainFTMT)
size(AlignedTrainFeatures)
[n_samples1, n_features] = size(FeatureTrainFTMT);
 [n_samples2, ~] = size(AlignedTrainFeatures);
if n_samples1 < n_samples2
        padding = n_samples2 - n_samples1; % Número de filas a agregar
        FeatureTrainFTMT = padarray(FeatureTrainFTMT, [padding, 0], 'post'); % Rellenar filas con ceros

        size(FeatureTrainFTMT)
        size(AlignedTrainFeatures)
    
    elseif n_samples2 < n_samples1
        padding = n_samples1 - n_samples2; % Número de filas a agregar
        AlignedTrainFeatures = padarray(AlignedTrainFeatures, [padding, 0], 'post'); % Rellenar filas con ceros
    end
FeatureTrain = [FeatureTrainFTMT; AlignedTrainFeatures];
FeatureTest = [FeatureTestFTMT; AlignedTestFeatures];

%% Train a Classifier
classifier = fitcecoc(FeatureTrain, YTrain);

%% Classify test images
YPred = predict(classifier, FeatureTest);

%% Evaluation
confMat = confusionmat(YTest, YPred);
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));  % Normalize confusion matrix
accuracy = mean(YPred == YTest);

disp('Accuracy:');
disp(accuracy);
disp('Confusion Matrix:');
disp(confMat);
