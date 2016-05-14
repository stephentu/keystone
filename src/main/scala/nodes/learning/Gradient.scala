package nodes.learning

import breeze.linalg._
import nodes.learning.Gradient.{SparseGradient, DenseGradient}
import utils.MatrixUtils

/**
 * Computes a gradient given vectors as data, labels, and model weights
 */
trait Gradient[T <: Vector[Double]] extends Serializable {

  def compute(
    numFeatures: Int,
    numClasses: Int,
    data: Iterator[T],
    labels: Iterator[DenseVector[Double]],
    weights: DenseMatrix[Double])
  : (DenseMatrix[Double], Double)
}

object Gradient {
  type DenseGradient = Gradient[DenseVector[Double]]
  type SparseGradient = Gradient[SparseVector[Double]]
}

/**
 * Computes a Least Squares loss gradient given [[DenseVector]]s as data
 */
class LeastSquaresDenseGradient extends DenseGradient {

  def compute(
    numFeatures: Int,
    numClasses: Int,
    data: Iterator[DenseVector[Double]],
    labels: Iterator[DenseVector[Double]],
    weights: DenseMatrix[Double])
  : (DenseMatrix[Double], Double) = {
    if (data.hasNext && labels.hasNext) {
      val dataMatrix = MatrixUtils.rowsToMatrix(data)
      val labelsMatrix = MatrixUtils.rowsToMatrix(labels)

      // Least Squares Gradient is At.(Ax - b)
      val axb = dataMatrix * weights - labelsMatrix
      val grad = dataMatrix.t * (axb)
      // Loss is 0.5 * norm(Ax - b)
      val loss = 0.5 * math.pow(norm(axb.toDenseVector), 2)

      (grad, loss)
    } else {
      (DenseMatrix.zeros[Double](numFeatures, numClasses), 0)
    }
  }
}

/**
 * Computes a Least Squares loss gradient given [[SparseVector]]s as data
 */
class LeastSquaresSparseGradient extends SparseGradient {
  override def compute(
    numFeatures: Int,
    numClasses: Int,
    data: Iterator[SparseVector[Double]],
    labels: Iterator[DenseVector[Double]],
    weights: DenseMatrix[Double])
  : (DenseMatrix[Double], Double) = {
    val gradient = DenseMatrix.zeros[Double](numFeatures, numClasses)
    var loss = 0.0

    while (data.hasNext) {
      val feature = data.next()
      val label = labels.next()

      // Least Squares Gradient is At.(Ax - b)
      val axb = weights.t * feature
      axb -= label

      if (label.length == 1) {
        // Performance optimization for the binary case
        // Data is  dx1

        // axb is 0
        var axb = 0.0

        var offset = 0
        while(offset < feature.activeSize) {
          val index = feature.indexAt(offset)
          val value = feature.valueAt(offset)
          axb += weights.data(index) * value
          offset += 1
        }

        axb -= label(0)

        offset = 0
        while(offset < feature.activeSize) {
          val index = feature.indexAt(offset)
          val value = feature.valueAt(offset)
          val gradUpdate = (axb * value)
          gradient(index, 0) += gradUpdate
          offset += 1
        }

        loss = loss + 0.5 * axb * axb

      } else {
        var offset = 0
        while (offset < feature.activeSize) {
          val index = feature.indexAt(offset)
          val value = feature.valueAt(offset)
          gradient(index, ::) += (axb.t * value)
          offset += 1
        }
        loss = loss + 0.5 * math.pow(norm(axb), 2)
      }
    }

    (gradient, loss)
  }
}


trait BatchGradient extends Serializable {
  
  def compute(
      data: DenseMatrix[Double],
      dataColMeans: DenseVector[Double],
      dataColStdevs: Option[DenseVector[Double]],
      labels: DenseMatrix[Double],
      weights: DenseMatrix[Double])
    : (DenseMatrix[Double], Double)
}

class LeastSquaresBatchGradient extends BatchGradient {
  
  def compute(
      data: DenseMatrix[Double],
      dataColMeans: DenseVector[Double],
      dataColStdevs: Option[DenseVector[Double]],
      labels: DenseMatrix[Double],
      weights: DenseMatrix[Double])
    : (DenseMatrix[Double], Double) = {

    // Least Squares Gradient is At.(Ax - b)
    val axb = (data * weights)
    val meanTerm = (dataColMeans.t * weights).t
    axb(*, ::) -= meanTerm
    dataColStdevs.foreach { x =>
      axb(*, ::) :/= x
    }
    axb -= labels

    val grad = data.t * (axb)
    // Loss is 0.5 * norm(Ax - b)
    val loss = 0.5 * math.pow(norm(axb.toDenseVector), 2)

    (grad, loss)
  }

}
