package utils

import org.scalatest.FunSuite

class ImageUtilsSuite extends FunSuite {

  test("crop") {
    val imgArr =
      (0 until 4).flatMap { x =>
        (0 until 4).flatMap { y =>
          (0 until 1).map { c =>
            (c + x * 1 + y * 4 * 1).toDouble
          }
        }
      }.toArray

    val image = new ChannelMajorArrayVectorizedImage(imgArr, ImageMetadata(4, 4, 1))
    val cropped = ImageUtils.crop(image, 1, 1, 3, 3)

    assert(cropped.metadata.xDim == 2)
    assert(cropped.metadata.yDim == 2)
    assert(cropped.metadata.numChannels == 1)

    assert(cropped.get(0, 0, 0) == 5.0)
    assert(cropped.get(0, 1, 0) == 6.0)
    assert(cropped.get(1, 0, 0) == 9.0)
    assert(cropped.get(1, 1, 0) == 10.0)
  }

}
