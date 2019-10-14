//
//  DataSource.m
//  espcnnproj
//
//  Created by garryling on 2019/10/9.
//  Copyright © 2019 garryling. All rights reserved.
//

#import "DataSource.h"

@interface DataSource(){
    NSString *_name;
    int _kernelWidth;
    int _kernelHeight;
    int _inputChannels;
    int _outputChannels;
    BOOL _issubpixelLayer;
    NSString *_biasptah;
    NSString *_weightpath;
}
@property (nonatomic, strong) NSData *weightData;
@property (nonatomic, strong) NSData *biasData;
@end

@implementation DataSource

-(instancetype) initWithName:(NSString *)name kernelWidth:(int)kw kernelHeight:(int)kh inputFeatureChannels:(int)inputChannels outputChannels:(int)outputChannels isSubpixelLayer:(BOOL)isSubpixelLayer{
    self = [super init];
    if (self) {
        _name = name;
        _kernelWidth = kw;
        _kernelHeight = kh;
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _issubpixelLayer = isSubpixelLayer;
    }
    return self;
}
    
-(BOOL) load{
    _weightpath = [[NSBundle mainBundle] pathForResource:[_name stringByAppendingString:@".weight.bin"] ofType:@""];
    _biasptah = [[NSBundle mainBundle] pathForResource:[_name stringByAppendingString:@".bias.bin"] ofType:@""];
    if ([[NSFileManager defaultManager] fileExistsAtPath:_weightpath] && [[NSFileManager defaultManager] fileExistsAtPath:_biasptah]) {

        _weightData = [[NSData alloc] initWithContentsOfFile:_weightpath];
        _biasData = [[NSData alloc] initWithContentsOfFile:_biasptah];
        return YES;
    }
    if (_issubpixelLayer) {
        return YES;
    }
    return NO;
}

- (MPSCNNConvolutionDescriptor *)descriptor{
    MPSCNNConvolutionDescriptor *bdesc;
    if (_issubpixelLayer) {
        bdesc = [MPSCNNSubPixelConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:_kernelWidth kernelHeight:_kernelHeight inputFeatureChannels:_inputChannels outputFeatureChannels:_outputChannels];
    
        MPSCNNSubPixelConvolutionDescriptor *subpixelDesc = (MPSCNNSubPixelConvolutionDescriptor *)bdesc;
        subpixelDesc.subPixelScaleFactor = 3;
    } else {
        bdesc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:_kernelWidth kernelHeight:_kernelHeight inputFeatureChannels:_inputChannels outputFeatureChannels:_outputChannels];
        [bdesc setNeuronType:MPSCNNNeuronTypeReLU parameterA:0 parameterB:0];
    }
    bdesc.strideInPixelsX = 1;
    bdesc.strideInPixelsY = 1;
    return bdesc;
}

-(void * __nonnull) weights{
    return (void *)_weightData.bytes;
}

-(float * __nullable) biasTerms{
    return (float *)_biasData.bytes;
}

- (MPSDataType)dataType{
    return MPSDataTypeFloat32;
}

-(void) purge{
    _biasData = nil;
    _weightData = nil;
}

-(NSString*__nullable) label{
    return _name;
}

- (nonnull id)copyWithZone:(nullable NSZone *)zone {
    assert(0);
    return nil;
}

@end
