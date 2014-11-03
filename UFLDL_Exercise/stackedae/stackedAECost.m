function [ cost, grad ] = stackedAECost(stackedAETheta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% softmaxTheta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(stackedAETheta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(stackedAETheta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

%FeedForward

stacklayer=cell(size(stack));
for d=1:numel(stack)
    if d==1
        stacklayer{d}.layerin=stack{d}.w*data+repmat(stack{d}.b,1,M);
    else
        stacklayer{d}.layerin=stack{d}.w*stacklayer{d-1}.layerout+...
            repmat(stack{d}.b,1,M);
    end
    stacklayer{d}.layerout=sigmoid(stacklayer{d}.layerin);
end

htheta=exp(softmaxTheta*stacklayer{numel(stack)}.layerout);
ptheta=htheta./repmat(sum(htheta),size(softmaxTheta,1),1);
cost=-1/M*sum(sum(groundTruth.*log(ptheta)))+...
    lambda/2*sum(sum(softmaxTheta.*softmaxTheta));

%BackPropogation
delta=cell(size(stack));

softmaxThetaGrad=lambda*softmaxTheta-1/M*...
    ((groundTruth-ptheta)*stacklayer{numel(stack)}.layerout');

for d=numel(stack):-1:1
    if d==numel(stack)
        delta{d}=-softmaxTheta'*(groundTruth-ptheta).*stacklayer{d}.layerout.*...
            (1-stacklayer{d}.layerout);
    else
        delta{d}=(stack{d+1}.w'*delta{d+1}).*stacklayer{d}.layerout.*...
            (1-stacklayer{d}.layerout);
    end
    if d==1
        stackgrad{d}.w=1/M*delta{d}*data';
    else
        stackgrad{d}.w=1/M*delta{d}*stacklayer{d-1}.layerout';
    end
    stackgrad{d}.b=1/M*sum(delta{d},2);
end

depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1, 1);
a{1} = data;

for layer = (1:depth)
  z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
  a{layer+1} = sigmoid(z{layer+1});
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
