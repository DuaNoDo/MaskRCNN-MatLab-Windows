function net = createMaskRCNN(lgraph,numClasses,params) 
% Create Mask R-CNN network from Faster R-CNN network specified by lgraph

% Copyright 2020 The MathWorks, Inc.

% Replace the input layer
inputLayer = imageInputLayer(params.ImageSize,'Normalization','rescale-symmetric', ...
    'Max',255,'Min',0,'name','input');
lgraph = replaceLayer(lgraph,'data',inputLayer);

% Drop loss layers
lgraph = removeLayers(lgraph,lgraph.OutputNames);

% Swap RPN softmax with the custom layer
rpnSftmax = layer.RPNSoftmax('rpnSoftmax');
lgraph = replaceLayer(lgraph,'rpnSoftmax',rpnSftmax);

% Add mask head to Faster R-CNN
maskHead = createMaskHead(numClasses,params);
lgraph = addLayers(lgraph,maskHead);
lgraph = connectLayers(lgraph,'res5c_relu','mask_tConv1');

% Replace RegionProposalLayer with custom RPL
customRegionProposal = layer.RegionProposal('rpl',params.AnchorBoxes,params);
lgraph = replaceLayer(lgraph,'regionProposal',customRegionProposal);

% Replace roiMaxpooling with roiAlign
roiAlign = roiAlignLayer([14 14],'Name','roiAlign','ROIScale',params.ScaleFactor(1));
lgraph = replaceLayer(lgraph,'roiPooling',roiAlign);

net = lgraph;
    
end
    
    
function layers = createMaskHead(numClasses,params)

    if(params.ClassAgnosticMasks)
        numMaskClasses = 1;
    else
        numMaskClasses = numClasses;
    end

    tconv1 = transposedConv2dLayer(2,256,'Stride',2,'Name','mask_tConv1');
    conv1 = convolution2dLayer(1,numMaskClasses,'Name','mask_Conv1', ...
        'Padding','same');
    sig1 = sigmoidLayer('Name','mask_sigmoid1');

    layers = [tconv1 conv1 sig1];  
end

