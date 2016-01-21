package nodes.images

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD
import pipelines.FunctionNode
import utils.{ImageMetadata, ChannelMajorArrayVectorizedImage, Image}


/**
 * @param numPatches Number of random patches to create for each image.
 * @param windowDim Dimension of the window (24 for CIFAR)
 */
class RandomFlips(numPatches: Int) extends FunctionNode[RDD[Image], RDD[Image]] {

  def apply(in: RDD[Image]) = {
    in.flatMap(getImageWindow)
  }

  //This function will take in an image and return `numPatches` 24x24 random
  //patches from the original image, as well as their horizontal flips
  def getCroppedImages(image: Image) = {
    val xDim = image.metadata.xDim
    val yDim = image.metadata.yDim
    val numChannels = image.metadata.numChannels

    (0 until numPatches).flatMap { x =>
      val randomPatch = new DenseVector[Double](windowDim * windowDim * numChannels)
      //Assume x and y border are same size, for CIFAR, borderSize=8=32-24
      val borderSize = xDim - windowDim
      // Pick a random int between 0 (inclusive) and borderSize (exclusive)
      val r = new scala.util.Random(123L)
      val startX = r.nextInt(borderSize)
      val endX = startX + windowDim
      val startY = r.nextInt(borderSize)
      val endY = startY + windowDim
      var c = 0
      while (c < numChannels) {
        var s = startX
        while (s < endX) {
          var b = startY
          while (b < endY) {
            pool(c + (s-startX)*numChannels +
              (b-startY)*(endX-startX)*numChannels) = image.get(s, b, c)
            b = b + 1
          }
          s = s + 1
        }
        c = c + 1
      }
      ChannelMajorArrayVectorizedImage(pool.toArray,
        ImageMetadata(windowSize, windowSize, numChannels))
    }
  }

}
