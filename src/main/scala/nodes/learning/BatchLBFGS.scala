/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package nodes.learning

import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.stats._
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}

import edu.berkeley.cs.amplab.mlmatrix.util.{Utils => MLMatrixUtils}

import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast

import workflow.LabelEstimator
import nodes.stats.{StandardScaler, StandardScalerModel}
import pipelines.Logging
import utils.{MatrixUtils, Stats}

/**
 * :: DeveloperApi ::
 * Class used to solve an optimization problem using Limited-memory BFGS.
 * Reference: [[http://en.wikipedia.org/wiki/Limited-memory_BFGS]]
 *
 * @param gradient Gradient function to be used.
 * @param numCorrections 3 < numCorrections < 10 is recommended.
 * @param convergenceTol convergence tolerance for L-BFGS
 * @param regParam L2 regularization
 * @param numIterations max number of iterations to run 
 */
class BatchLBFGSwithL2(
    val gradient: BatchGradient, 
    val numCorrections: Int  = 10,
    val convergenceTol: Double = 1e-4,
    val numIterations: Int = 100,
    val regParam: Double = 0.0,
    val epochCallback: Option[LinearMapper[DenseVector[Double]] => Double] = None,
    val epochEveryTest: Int = 10)
  extends LabelEstimator[DenseVector[Double], DenseVector[Double], DenseVector[Double]] {

  def fit(data: RDD[DenseVector[Double]], labels: RDD[DenseVector[Double]]): LinearMapper[DenseVector[Double]] = {
    val out = fitBatch(data.mapPartitions(itr => MatrixUtils.rowsToMatrixIter(itr)),
      labels.mapPartitions(itr => MatrixUtils.rowsToMatrixIter(itr)))
    LinearMapper(out._1, out._2, out._3)
  }

  def fitBatch(
      data: RDD[DenseMatrix[Double]],
      labels: RDD[DenseMatrix[Double]])
    : (DenseMatrix[Double], Option[DenseVector[Double]], Option[StandardScalerModel]) = {

    val numExamples = data.map(_.rows).reduce(_ + _)
    val numFeatures = data.map(_.cols).collect().head
    val numClasses = labels.map(_.cols).collect().head

		// TODO: Do stdev division ?
    val popFeatureMean = BatchLBFGSwithL2.computeColMean(data, numExamples, numFeatures)
		val featureScaler = new StandardScalerModel(popFeatureMean, None)
		val labelScaler = new StandardScalerModel(BatchLBFGSwithL2.computeColMean(labels, numExamples, numClasses), None)
    // val featureScaler = new StandardScaler(normalizeStdDev = false).fit(data)
    // val labelScaler = new StandardScaler(normalizeStdDev = false).fit(labels)

    val model = BatchLBFGSwithL2.runLBFGS(
      data,
      labels,
      featureScaler,
      labelScaler,
      gradient,
      numCorrections,
      convergenceTol,
      numIterations,
      regParam,
      epochCallback,
      epochEveryTest)

    (model, Some(labelScaler.mean), Some(featureScaler))
  }

}

object BatchLBFGSwithL2 extends Logging {

	def powInPlace(in: DenseVector[Double], power: Double) = {
		var i = 0
		while (i < in.size) {
			in(i) = math.pow(in(i), power)
			i = i + 1
		}
		in
	}

  def computeColMean(
    data: RDD[DenseMatrix[Double]],
    nRows: Long,
    nCols: Int): DenseVector[Double] = {
    // To compute the column means, compute the colSum in each partition, add it
    // up and then divide by number of rows.
    data.aggregate(DenseVector.zeros[Double](nCols))(
			seqOp = (a: DenseVector[Double], b: DenseMatrix[Double]) => {
	      a += sum(b(::, *)).toDenseVector
			},
			combOp = (a: DenseVector[Double], b: DenseVector[Double]) => a += b
    ) /= nRows.toDouble
  }

  /**
   * Run Limited-memory BFGS (L-BFGS) in parallel.
   * Averaging the subgradients over different partitions is performed using one standard
   * spark map-reduce in each iteration.
   */
  def runLBFGS(
      data: RDD[DenseMatrix[Double]],
      labels: RDD[DenseMatrix[Double]],
      featureScaler: StandardScalerModel,
      labelScaler: StandardScalerModel,
      gradient: BatchGradient,
      numCorrections: Int,
      convergenceTol: Double,
      maxNumIterations: Int,
      regParam: Double,
      epochCallback: Option[LinearMapper[DenseVector[Double]] => Double] = None,
      epochEveryTest: Int = 10): DenseMatrix[Double] = {

    val lossHistory = mutable.ArrayBuilder.make[Double]
    val numExamples = data.map(_.rows).reduce(_ + _)
    val numFeatures = data.map(_.cols).collect().head
    val numClasses = labels.map(_.cols).collect().head

    val startConversionTime = System.currentTimeMillis()

		// val dataMat = data.map { x =>
		// 	x(*, ::) -= featureScaler.mean
		// }

    //val dataMat = featureScaler.apply(data).mapPartitions { part =>
    //  Iterator.single(MatrixUtils.rowsToMatrix(part))
    //}.persist(StorageLevel.MEMORY_AND_DISK)

    val labelsMat = labels.map { x =>
			x(*, ::) - labelScaler.mean
		}.cache()
		labelsMat.count	
		//  labelScaler.apply(labels).mapPartitions { part =>
    //  Iterator.single(MatrixUtils.rowsToMatrix(part))
    //}.persist(StorageLevel.MEMORY_AND_DISK)

    //dataMat.count()
    //labelsMat.count()

    //data.unpersist()
    //labels.unpersist()
    val endConversionTime = System.currentTimeMillis()
    logInfo(s"PIPELINE TIMING: Finished System Conversion And Transfer in ${endConversionTime - startConversionTime} ms")

    val costFun = new CostFun(data, featureScaler.mean, labelsMat, gradient, regParam, numExamples, numFeatures,
      numClasses)

    val lbfgs = new BreezeLBFGS[DenseVector[Double]](maxNumIterations, numCorrections, convergenceTol)

    val initialWeights = DenseVector.zeros[Double](numFeatures * numClasses)

    val states =
      lbfgs.iterations(new CachedDiffFunction(costFun), initialWeights)

    /**
     * NOTE: lossSum and loss is computed using the weights from the previous iteration
     * and regVal is the regularization value computed in the previous iteration as well.
     */
    var epoch = 0
    var state = states.next()
    while (states.hasNext) {
      val epochBegin = System.nanoTime
      lossHistory += state.value
      state = states.next()
      println("For epoch " + epoch + " value is " + state.value)
      println("For epoch " + epoch + " iter is " + state.iter)
      println("For epoch " + epoch + " grad norm is " + norm(state.grad))
      println("For epoch " + epoch + " searchFailed ? " + state.searchFailed)
      println("For epoch " + epoch + " x norm " + norm(state.x))
      val epochTime = System.nanoTime - epochBegin
      println("EPOCH_" + epoch + "_time: " + epochTime)
      if (!epochCallback.isEmpty && (epochEveryTest == 1 || epoch % epochEveryTest == 1)) {
        val weights = state.x.asDenseMatrix.reshape(numFeatures, numClasses)
        val lm = LinearMapper[DenseVector[Double]](weights, Some(labelScaler.mean), Some(featureScaler))
        val testAcc = epochCallback.get(lm)
        println(s"EPOCH_${epoch}_LAMBDA_${regParam}_TEST_ACC_${testAcc}")
        //println("For epoch " + epoch + " TEST accuracy " + epochCallback.get(lm))
      }

			epoch = epoch + 1
    }
    lossHistory += state.value
    val finalWeights = state.x.asDenseMatrix.reshape(numFeatures, numClasses)

    val lossHistoryArray = lossHistory.result()

    logInfo("LBFGS.runLBFGS finished. Last 10 losses %s".format(
      lossHistoryArray.takeRight(10).mkString(", ")))

    finalWeights
  }

  /**
   * CostFun implements Breeze's DiffFunction[T], which returns the loss and gradient
   * at a particular point (weights). It's used in Breeze's convex optimization routines.
   */
  private class CostFun(
    dataMat: RDD[DenseMatrix[Double]],
		dataColMeans: DenseVector[Double],
    labelsMat: RDD[DenseMatrix[Double]],
    gradient: BatchGradient,
    regParam: Double,
    numExamples: Long,
    numFeatures: Int,
    numClasses: Int) extends DiffFunction[DenseVector[Double]] {

    override def calculate(weights: DenseVector[Double]): (Double, DenseVector[Double]) = {
      val weightsMat = weights.asDenseMatrix.reshape(numFeatures, numClasses)
      // Have a local copy to avoid the serialization of CostFun object which is not serializable.
      val bcW = dataMat.context.broadcast(weightsMat)
			val localColMeansBC = dataMat.context.broadcast(dataColMeans)
      val localGradient = gradient

      val gradResult = MLMatrixUtils.treeReduce(dataMat.zip(labelsMat).map { x =>
          localGradient.compute(x._1, localColMeansBC.value, x._2, bcW.value)
        }, (a: GradientResult, b: GradientResult) => { 
          a.gradient += b.gradient
          GradientResult(a.gradient, a.loss + b.loss) 
        }
      )
      val gradientSum = gradResult.gradient
      val lossSum = gradResult.loss

      // total loss = lossSum / nTrain + 1/2 * lambda * norm(W)^2
      val normWSquared = math.pow(norm(weights), 2)
      val regVal = 0.5 * regParam * normWSquared
      val loss = lossSum / numExamples + regVal

      localColMeansBC.destroy()
      bcW.destroy()

      // total gradient = gradSum / nTrain + lambda * w
      val gradientTotal = gradientSum / numExamples.toDouble + (weightsMat * regParam)

      (loss, gradientTotal.toDenseVector)
    }
  }
}
