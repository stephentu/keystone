package nodes.stats

import breeze.linalg._
import breeze.numerics._
import workflow.Transformer
import utils.MatrixUtils

case class GaussianKernel(gamma: Double) {

  private def expScalaMulInPlace(pdist2: DenseMatrix[Double], gamma: Double) = {
    var j = 0
    // go in a column major order ?
    while (j < pdist2.cols) {
      var i = 0
      while (i < pdist2.rows) {
        pdist2(i, j) *= -gamma
        pdist2(i, j) = math.exp(pdist2(i, j))
        i = i + 1
      }
      j = j + 1
    }
    pdist2
  }

  def apply(in: DenseMatrix[Double]): DenseMatrix[Double] = {
    val pdist2 = MatrixUtils.squaredPDist(in)
    expScalaMulInPlace(pdist2, gamma)
    pdist2
  }

  def apply(lhs: DenseMatrix[Double], rhs: DenseMatrix[Double]): DenseMatrix[Double] = {
    val pdist2 = MatrixUtils.squaredPDist(lhs, rhs)
    expScalaMulInPlace(pdist2, gamma)
    pdist2
  }
}
