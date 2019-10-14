//
//  ViewController.m
//  espcnnproj
//
//  Created by garryling on 2019/10/9.
//  Copyright © 2019 garryling. All rights reserved.
//
#import <opencv2/opencv.hpp>
#import "ViewController.h"
#import "espcnnGraph.h"
#import <MetalKit/MetalKit.h>

static id<MTLDevice> g_device;

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    g_device =  MTLCreateSystemDefaultDevice();
    espcnnGraph *graph = [[espcnnGraph alloc] initWithDevice:g_device];
    UIImage *testImage = [UIImage imageNamed:@"test.jpg"];
    cv::Mat inputmat = [self cvMatFromUIImage:testImage];
    cv::Mat outputmat;
    cv::cvtColor(inputmat, outputmat, cv::COLOR_RGB2YCrCb);
    cv::Mat channel[3];
    cv::split(outputmat, channel);
    cv::Mat ychannel = channel[0];
    cv::Mat crchannel = channel[1];
    cv::Mat cbchannel = channel[2];
    cv::Mat ychannelNorm;
    ychannel.convertTo(ychannelNorm, CV_32FC1, 1.0f / 255.0f);
    MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                                 width:ychannelNorm.cols height:ychannelNorm.rows mipmapped:NO];
    NSData *wrapData = [[NSData alloc] initWithBytes:ychannelNorm.data length:ychannelNorm.cols * ychannelNorm.rows * 4];
    id<MTLTexture> testTexture = [g_device newTextureWithDescriptor:textureDescriptor];
    
    NSUInteger bytesPerRow = 4 * textureDescriptor.width;
    
    MTLRegion region = {{0,0,0}, {textureDescriptor.width, textureDescriptor.height, 1}};
    
    [testTexture replaceRegion:region mipmapLevel:0 withBytes:wrapData.bytes bytesPerRow:bytesPerRow];
    
    [graph prediction:testTexture successBlock:^(MPSImage * _Nonnull resultMPSImage) {
        MTLRegion targetRegion = {{0, 0, 0}, {resultMPSImage.width, resultMPSImage.height, 1}};
        float16_t *dstData = (float16_t *)malloc(resultMPSImage.width * resultMPSImage.height * resultMPSImage.featureChannels * 2);
        [resultMPSImage.texture getBytes:dstData bytesPerRow:resultMPSImage.width * 2 fromRegion:targetRegion mipmapLevel:0];
        cv::Mat targetMat = cv::Mat((int)resultMPSImage.height, (int)resultMPSImage.width, CV_16FC1, dstData);
//        targetMat.convertTo(targetMat, CV_32FC1);
//        cv::threshold(targetMat, targetMat, 1, 0, cv::THRESH_TRUNC);
//        cv::threshold(targetMat, targetMat, 0, 0, cv::THRESH_TOZERO);
//        std::cout << targetMat << std::endl;
        cv::Mat targetIntMat;
        targetMat.convertTo(targetIntMat, CV_8UC1, 255.0f);
        UIImage *targetImg = [self UIImageFromCVMat:targetIntMat];
//        std::cout << targetIntMat << std::endl;
    }];
}


- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
  CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
  CGFloat cols = image.size.width;
  CGFloat rows = image.size.height;
  cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
  CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                 cols,                       // Width of bitmap
                                                 rows,                       // Height of bitmap
                                                 8,                          // Bits per component
                                                 cvMat.step[0],              // Bytes per row
                                                 colorSpace,                 // Colorspace
                                                 kCGImageAlphaNoneSkipLast |
                                                 kCGBitmapByteOrderDefault); // Bitmap info flags
  CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
  CGContextRelease(contextRef);
  return cvMat;
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
  NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
  CGColorSpaceRef colorSpace;

  if (cvMat.elemSize() == 1) {
      colorSpace = CGColorSpaceCreateDeviceGray();
  } else {
      colorSpace = CGColorSpaceCreateDeviceRGB();
  }

    if (cvMat.isContinuous()) {
        
    }
  CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);

  // Creating CGImage from cv::Mat
  CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                     cvMat.rows,                                 //height
                                     8,                                          //bits per component
                                     8 * cvMat.elemSize(),                       //bits per pixel
                                     cvMat.step[0],                            //bytesPerRow
                                     colorSpace,                                 //colorspace
                                     kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                     provider,                                   //CGDataProviderRef
                                     NULL,                                       //decode
                                     false,                                      //should interpolate
                                     kCGRenderingIntentDefault                   //intent
                                     );


  // Getting UIImage from CGImage
  UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
  CGImageRelease(imageRef);
  CGDataProviderRelease(provider);
  CGColorSpaceRelease(colorSpace);

  return finalImage;
 }

@end
