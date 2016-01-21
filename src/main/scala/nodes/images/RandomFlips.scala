package nodes.images

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD
import pipelines.FunctionNode
import utils.{ImageMetadata, ChannelMajorArrayVectorizedImage, Image}


/**
 * @param numPatches Number of random patches to create for each image.
 * @param windowDim Dimension of the window (24 for CIFAR)
 */
class RandomFlips(
    numPatches: Int,
    windowDim: Int,
    centerCorners: Boolean = false) extends FunctionNode[RDD[Image], RDD[Image]] {

  def apply(in: RDD[Image]) = {
    if (centerCorners) {
      in.flatMap(getCroppedTestImages)
    } else {
      in.flatMap(getCroppedImages)
    }
  }

  // This function will take in an image and return `numPatches` 24x24 random
  // patches from the original image, as well as their horizontal flips
  def getCroppedImages(image: Image) = {
    val xDim = image.metadata.xDim
    val yDim = image.metadata.yDim
    val numChannels = image.metadata.numChannels

    val r = new scala.util.Random(123L)
    (0 until numPatches).map { x =>
      // Flip with 50% probability as in cuda-convnet
      val flip = r.nextBoolean()
      val randomPatch = new DenseVector[Double](windowDim * windowDim * numChannels)
      // Assume x and y border are same size, for CIFAR, borderSize=8=32-24
      val borderSize = xDim - windowDim
      // Pick a random int between 0 and borderSize (inclusive)
      val startX = r.nextInt(borderSize + 1)
      val endX = startX + windowDim
      val startY = r.nextInt(borderSize + 1)
      val endY = startY + windowDim
      var c = 0
      while (c < numChannels) {
        var s = startX
        while (s < endX) {
          if (!flip) {
            var b = startY
            while (b < endY) {
              randomPatch(c + (s-startX)*numChannels +
                (b-startY)*(endX-startX)*numChannels) = image.get(s, b, c)
              b = b + 1
            }
          } else {
            // TODO: Is this the right axis to flip ?
            var bSource = startY
            var bDest = endY - 1
            while (bDest >= startY) {
              randomPatch(c + (s-startX)*numChannels +
                (bDest-startY)*(endX-startX)*numChannels) = image.get(s, bSource, c)
              bDest = bDest - 1
              bSource = bSource + 1
            }
          }
          s = s + 1
        }
        c = c + 1
      }
      ChannelMajorArrayVectorizedImage(randomPatch.toArray,
        ImageMetadata(windowDim, windowDim, numChannels))
    }
  }
  /*
   * Get the 4 corner images as well as the center image of size windowDim x windowDim
   * and their flips
   */
  def getCroppedTestImages(image: Image) = {
    val xDim = image.metadata.xDim
    val yDim = image.metadata.yDim
    val numChannels = image.metadata.numChannels
    
    // Assume x is vertical axis and y is horizontal axis
    // These are the start indices for the upperLeft, lowerLeft, upperRight, lowerRight,
    // and center image (in that order)
    val startXs = Array(0, xDim-windowDim, 0, xDim-windowDim, (xDim-windowDim)/2) 
    val startYs = Array(0, 0, yDim-windowDim, yDim-windowDim, (yDim-windowDim)/2)

    (0 until startXs.length).flatMap { i =>
      val randomPatch = new DenseVector[Double](windowDim * windowDim * numChannels)
      val randomPatchFlipped = new DenseVector[Double](windowDim * windowDim * numChannels)
      val startX = startXs(i)
      val endX = startX + windowDim
      val startY = startYs(i)
      val endY = startY + windowDim
      var c = 0
      while (c < numChannels) {
        var s = startX
        while (s < endX) {
          var b = startY
          while (b < endY) {
            randomPatch(c + (s-startX)*numChannels +
              (b-startY)*(endX-startX)*numChannels) = image.get(s, b, c)
            b = b + 1
          }
          // TODO: Is this the right axis to flip ?
          var bSource = startY
          var bDest = endY - 1
          while (bDest >= startY) {
            randomPatchFlipped(c + (s-startX)*numChannels +
              (bDest-startY)*(endX-startX)*numChannels) = image.get(s, bSource, c)
            bDest = bDest - 1
            bSource = bSource + 1
          }
          s = s + 1
        }
        c = c + 1
      }
      Iterator(
        ChannelMajorArrayVectorizedImage(randomPatch.toArray,
          ImageMetadata(windowDim, windowDim, numChannels)),
        ChannelMajorArrayVectorizedImage(randomPatchFlipped.toArray,
          ImageMetadata(windowDim, windowDim, numChannels))
      )
    }
  }

}
