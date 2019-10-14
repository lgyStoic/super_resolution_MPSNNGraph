//
//  espcnnGraph.h
//  espcnnproj
//
//  Created by garryling on 2019/10/9.
//  Copyright © 2019 garryling. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

NS_ASSUME_NONNULL_BEGIN

@interface espcnnGraph : NSObject
-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) inputDevice;
-(void) prediction:(id<MTLTexture>) inputTexture successBlock:(void(^)(MPSImage *))resultBlock;
@end
NS_ASSUME_NONNULL_END
