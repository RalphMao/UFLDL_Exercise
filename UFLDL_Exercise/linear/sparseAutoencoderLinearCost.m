function [cost,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    


W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);


% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));


datasize=size(data,2);

layer2_in=W1*data+repmat(b1,1,datasize);
layer2_out=sigmoid(layer2_in);
layer3_in=W2*layer2_out+repmat(b2,1,datasize);
layer3_out=layer3_in;

cost=sum(sum((layer3_out-data).^2))/2/datasize+lambda/2*(sum(sum(W1.^2))+sum(sum((W2.^2))));
ro_d=sum(layer2_out,2)/datasize;
cost=cost+beta*sum(sparsityParam*log(sparsityParam./ro_d)+(1-sparsityParam)*log((1-sparsityParam)./(1-ro_d)));

delta2=(layer3_out-data);

W2grad=1/datasize*delta2*layer2_out'+lambda*W2;
b2grad=1/datasize*sum(delta2,2);

delta=(W2'*delta2+repmat(beta*(-sparsityParam./ro_d+(1-sparsityParam)./(1-ro_d)),1,datasize)).*(layer2_out-layer2_out.^2);

W1grad=1/datasize*delta*data'+lambda*W1;
b1grad=1/datasize*sum(delta,2);

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end
function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
