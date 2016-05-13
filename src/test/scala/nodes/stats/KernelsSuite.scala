package nodes.stats

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import breeze.stats.distributions.{CauchyDistribution, Rand}
import org.scalatest.FunSuite
import utils.Stats
import utils.MatrixUtils

class KernelsSuite extends FunSuite {

  val gamma = 1.34
  val numEx = 100
  val numFeat = 20

  test("Guassian kernel works correctly") {
    val gk = GaussianKernel(gamma)
    val data = DenseMatrix.rand(numEx, numFeat)

    val kernelMat = gk(data)
    // kernel matrix should be square
    assert(kernelMat.rows == numEx)
    assert(kernelMat.cols == numEx)

    // test if our not inplace impl matches new impl
    val pdist2 = MatrixUtils.squaredPDist(data)
    val kernelMatExpected = exp(pdist2 * -gamma)

    assert(Stats.aboutEq(kernelMatExpected, kernelMat, 1e-6))
    val diff = (kernelMatExpected - kernelMat).toDenseVector
    // assert(diff.max < 1e-6)
  }

  test("Uniform partitioner works correctly") {
    val numElems = 10
    val numParts = 3
    val part = new UniformRangePartitioner(numElems, numParts)

    (0 until numElems).foreach { e =>
      println("Got partition " + part.getPartition(e) + " for " + e)
    }
  }

}
