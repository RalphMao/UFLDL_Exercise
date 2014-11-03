function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);
pooledsize=floor(convolvedDim / poolDim);
pooledFeatures=zeros(numFeatures,numImages,pooledsize,pooledsize);

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------
% convolvedFeatures=reshape(convolvedFeatures,numImages*numFeatures,convolvedDim.^2);
% convolvedFeatures=convolvedFeatures';
% convolvedFeatures=reshape(convolvedFeatures,poolDim*poolDim,pooledsize^2 ...
%     *numImages*numFeatures);
% 
% pooledFeatures=mean(convolvedFeatures);
% 
% pooledFeatures=reshape(pooledFeatures, pooledsize*pooledsize,numFeatures*numImages);
% pooledFeatures=pooledFeatures';
% pooledFeatures=reshape(pooledFeatures,numFeatures,numImages, pooledsize,pooledsize);

for i=1:numImages
    for j=1:numFeatures
        for x=1:pooledsize
            for y=1:pooledsize
               pooledFeatures(j,i,x,y)=mean(mean(convolvedFeatures(j,i,(x-1)*poolDim+1:x*poolDim,...
                   (y-1)*poolDim+1:y*poolDim)));
            end
        end
    end
end
end

