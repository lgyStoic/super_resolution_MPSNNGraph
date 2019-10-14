//
//  espcnnGraph.m
//  espcnnproj
//
//  Created by garryling on 2019/10/9.
//  Copyright © 2019 garryling. All rights reserved.
//

#import "espcnnGraph.h"
#import "DataSource.h"

const NSInteger upscale_factor = 3;

@interface espcnnGraph(){
    MPSNNImageNode *_inputImage;
    MPSCNNConvolutionNode *conv1;
    MPSCNNConvolutionNode *conv2;
    MPSCNNConvolutionNode *conv3;
    MPSCNNConvolutionNode *conv4;
    MPSNNGraph *inferenceGraph;
}
@end

@implementation espcnnGraph
-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) inputDevice{
    self = [super init];
    if (self) {
        
        _inputImage = [[MPSNNImageNode alloc] initWithHandle:nil];
        
        conv1 = [MPSCNNConvolutionNode nodeWithSource:_inputImage
                                              weights:[[DataSource alloc] initWithName:@"conv1"
                                                                           kernelWidth:5
                                                                          kernelHeight:5
                                                                  inputFeatureChannels:1
                                                                        outputChannels:64
                                                                       isSubpixelLayer:NO]];
        MPSCNNNeuronReLUNode *r1 = [MPSCNNNeuronReLUNode nodeWithSource:conv1.resultImage];
        conv2 = [MPSCNNConvolutionNode nodeWithSource:r1.resultImage
                                              weights:[[DataSource alloc] initWithName:@"conv2"
                                                                           kernelWidth:3
                                                                          kernelHeight:3
                                                                  inputFeatureChannels:64
                                                                        outputChannels:64
                                                                       isSubpixelLayer:NO]];
        MPSCNNNeuronReLUNode *r2 = [MPSCNNNeuronReLUNode nodeWithSource:conv2.resultImage];
        conv3 = [MPSCNNConvolutionNode nodeWithSource:r2.resultImage
                                              weights:[[DataSource alloc] initWithName:@"conv3"
                                                                           kernelWidth:3
                                                                          kernelHeight:3
                                                                  inputFeatureChannels:64
                                                                        outputChannels:32
                                                                       isSubpixelLayer:NO]];
        MPSCNNNeuronReLUNode *r3 = [MPSCNNNeuronReLUNode nodeWithSource:conv3.resultImage];
        conv4 = [MPSCNNConvolutionNode nodeWithSource:r3.resultImage
                                              weights:[[DataSource alloc] initWithName:@"conv4"
                                                                           kernelWidth:3
                                                                          kernelHeight:3
                                                                  inputFeatureChannels:32
                                                                        outputChannels:upscale_factor * upscale_factor
                                                                       isSubpixelLayer:YES]];
        inferenceGraph = [[MPSNNGraph alloc] initWithDevice:inputDevice resultImage:conv4.resultImage resultImageIsNeeded:YES];
        
    }
    return self;
}

-(void) prediction:(id<MTLTexture>) inputTexture successBlock:(void(^)(MPSImage *))resultBlock{
    MPSImage *inputImage = [[MPSImage alloc] initWithTexture:inputTexture featureChannels:1];
    [inferenceGraph executeAsyncWithSourceImages:@[inputImage] completionHandler:^(MPSImage * _Nullable result, NSError * _Nullable error) {
        resultBlock(result);
    }];
}

@end
