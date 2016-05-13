package nodes.stats

import breeze.linalg._
import breeze.numerics._
import workflow.Transformer
import utils.MatrixUtils

import org.apache.spark.Partitioner

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


// Uniformly partitions keys from 0 to n-1 into p partition
// TODO: Only works for Int keys right now
class UniformRangePartitioner(numElems: Int, numParts: Int) extends Partitioner {

  val elemsPerPart = math.ceil(numElems.toDouble/numParts).toInt

  def numPartitions: Int = numParts

  def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Int]
    k / elemsPerPart
    // (k.toDouble / numParts).toInt
  }

}

