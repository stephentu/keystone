package nodes.stats

import breeze.linalg._
import breeze.numerics._
import workflow.Transformer
import utils.MatrixUtils

case class GaussianKernel(gamma: Double) {
  def apply(in: DenseMatrix[Double]): DenseMatrix[Double] = {
    val pdist2 = MatrixUtils.squaredPDist(in)
    exp(pdist2 * (-gamma))
  }

  def apply(lhs: DenseMatrix[Double], rhs: DenseMatrix[Double]): DenseMatrix[Double] = {
    val pdist2 = MatrixUtils.squaredPDist(lhs, rhs)
    exp(pdist2 * (-gamma))
  }
}
