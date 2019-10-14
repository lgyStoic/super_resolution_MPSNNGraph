//
//  DataSource.h
//  espcnnproj
//
//  Created by garryling on 2019/10/9.
//  Copyright © 2019 garryling. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
NS_ASSUME_NONNULL_BEGIN

@interface DataSource : NSObject<MPSCNNConvolutionDataSource>
-(instancetype) initWithName:(NSString *)name kernelWidth:(int)kw kernelHeight:(int)kh inputFeatureChannels:(int)inputChannels outputChannels:(int)outputChannels isSubpixelLayer:(BOOL)isSubpixelLayer;
@end

NS_ASSUME_NONNULL_END
