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

case class SparseLinearKernel {

  private def sparseDotProduct(
    lhs: (Array[Long], Array[Double]),
    rhs: (Array[Long], Array[Double])): Double = {

		val lhsInds = lhs._1
		val lhsValues = lhs._2
		val rhsInds = rhs._1
		val rhsValues = rhs._2

    var s = 0.0
    var lhsIdx = 0
    var rhsIdx = 0
    while (lhsIdx < lhsInds.size && rhsIdx < rhsInds.size) {
      if (lhsInds(lhsIdx) == rhsInds(rhsIdx)) {
        s += lhsValues(lhsIdx) * rhsValues(rhsIdx)
        lhsIdx += 1
        rhsIdx += 1
      } else if (lhsInds(lhsIdx) < rhsInds(rhsIdx)) {
        lhsIdx += 1
      } else {
        rhsIdx += 1
      }
    }
    s
  }

	def apply(
			lhs: Seq[(Array[Long], Array[Double])],
			rhs: Seq[(Array[Long], Array[Double])]): DenseMatrix[Double] = {
		val lhsInput = lhs.size
		val rhsInput = rhs.size
		val out = DenseMatrix.zeros[Double](lhsInput, rhsInput)
		(0 until lhsInput).foreach { lhsIdx =>
			(0 until rhsInput).foreach { rhsIdx =>
        out(lhsIdx, rhsIdx) = sparseDotProduct(lhs(lhsIdx), rhs(rhsIdx))
			}
		}
		out
	}

  def apply(in: Seq[(Array[Long], Array[Double])]): DenseMatrix[Double] = {
		val numInput = in.length
		val out = DenseMatrix.zeros[Double](numInput, numInput)
		(0 until numInput).foreach { lhs =>
			(0 until numInput).foreach { rhs =>
				out(lhs, rhs) = sparseDotProduct(in(lhs), in(rhs))
			}
		}
		out
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

