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
 * Class used to solve an optimization problem using Mini-batch SGD.
 * Reference: [[http://en.wikipedia.org/wiki/Limited-memory_BFGS]]
 *
 * @param gradient Gradient function to be used.
 * @param convergenceTol convergence tolerance
 * @param regParam L2 regularization
 * @param numIterations max number of iterations to run 
 */
class MiniBatchSGDwithL2(
    val gradient: BatchGradient, 
    val convergenceTol: Double = 1e-4,
    val numIterations: Int = 100,
    val regParam: Double = 0.0,
    val stepSize: Double = 1.0,
    val miniBatchFraction: Double = 0.1,
    val normStd: Boolean = false,
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
    val popFeatureMean = MiniBatchSGDwithL2.computeColMean(data, numExamples, numFeatures)
    val popStdEv = if (normStd) {
      Some(MiniBatchSGDwithL2.computeColStdEv(data, numExamples, popFeatureMean, numFeatures))
    } else {
      None
    }

		val featureScaler = new StandardScalerModel(popFeatureMean, popStdEv)
		val labelScaler = new StandardScalerModel(MiniBatchSGDwithL2.computeColMean(labels, numExamples, numClasses), None)
    // val featureScaler = new StandardScaler(normalizeStdDev = false).fit(data)
    // val labelScaler = new StandardScaler(normalizeStdDev = false).fit(labels)

    val model = MiniBatchSGDwithL2.runSGD(
      data,
      labels,
      featureScaler,
      labelScaler,
      gradient,
      convergenceTol,
      numIterations,
      stepSize,
      miniBatchFraction,
      regParam,
      epochCallback,
      epochEveryTest)

    (model, Some(labelScaler.mean), Some(featureScaler))
  }

}

object MiniBatchSGDwithL2 extends Logging {

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

	def computeColStdEv(
	    data: RDD[DenseMatrix[Double]],
	    nRows: Long,
	 	  dataMean: DenseVector[Double],
	    nCols: Int): DenseVector[Double] = {
		val meanBC = data.context.broadcast(dataMean)
	  // To compute the std dev, compute (x - mean)^2 for each row and add it up 
	  // and then divide by number of rows.
		val variance = data.aggregate(DenseVector.zeros[Double](nCols))(
			seqOp = (a: DenseVector[Double], b: DenseMatrix[Double]) => {
				var i = 0
				val mean = meanBC.value
				while (i < b.rows) {
				  val diff = (b(i, ::).t - mean)
					powInPlace(diff, 2.0)
					a += diff
					i = i + 1
				}
				a
		  },
			combOp = (a: DenseVector[Double], b: DenseVector[Double]) => a += b) 
		variance /= (nRows.toDouble - 1.0)
		powInPlace(variance, 0.5)
    variance
	}


  /**
   * Run Limited-memory BFGS (L-BFGS) in parallel.
   * Averaging the subgradients over different partitions is performed using one standard
   * spark map-reduce in each iteration.
   */
  def runSGD(
      data: RDD[DenseMatrix[Double]],
      labels: RDD[DenseMatrix[Double]],
      featureScaler: StandardScalerModel,
      labelScaler: StandardScalerModel,
      gradient: BatchGradient,
      convergenceTol: Double,
      maxNumIterations: Int,
      stepSize: Double,
      miniBatchFraction: Double,
      regParam: Double,
      epochCallback: Option[LinearMapper[DenseVector[Double]] => Double] = None,
      epochEveryTest: Int = 10): DenseMatrix[Double] = {

    val lossHistory = mutable.ArrayBuilder.make[Double]
    val numExamples = data.map(_.rows).reduce(_ + _)
    val numFeatures = data.map(_.cols).collect().head
    val numClasses = labels.map(_.cols).collect().head

    val startConversionTime = System.currentTimeMillis()

    val labelsMat = labels.map { x =>
			x(*, ::) - labelScaler.mean
		}.cache()
		labelsMat.count	
    val endConversionTime = System.currentTimeMillis()
    logInfo(s"PIPELINE TIMING: Finished System Conversion And Transfer in ${endConversionTime - startConversionTime} ms")

    val gradFun = new GradFun(data, featureScaler.mean, featureScaler.std, labelsMat, gradient, regParam, 
      miniBatchFraction, numExamples, numFeatures, numClasses)

    val initialWeights = DenseVector.zeros[Double](numFeatures * numClasses)

    /**
     * NOTE: lossSum and loss is computed using the weights from the previous iteration
     * and regVal is the regularization value computed in the previous iteration as well.
     */
    var prevWeights = null
    var currentWeights = initialWeights
    var currentLoss = 0.0
    var epoch = 0
    while (epoch < maxNumIterations && !isConverged(prevWeights, currentWeights, convergenceTol)) {
      val epochBegin = System.nanoTime
      val (loss, gradient) = gradFun.calculate(currentWeights)

      lossHistory += loss
      val prevWeights = currentWeights

      // TODO: This uses sqrt(stepSize) policy, we can change this if reqd
      val thisIterStepSize = stepSize / math.sqrt(epoch)
      currentWeights = prevWeights * (1.0 - thisIterStepSize * regParam)
      currentWeights -= thisIterStepSize * gradient 

      println("For epoch " + epoch + " loss is " + loss)
      println("For epoch " + epoch + " grad norm is " + norm(gradient))
      println("For epoch " + epoch + " x norm " + norm(prevWeights))
      val epochTime = System.nanoTime - epochBegin
      println("EPOCH_" + epoch + "_time: " + epochTime)
      if (!epochCallback.isEmpty && epoch % epochEveryTest == 1) {
        val weights = currentWeights.asDenseMatrix.reshape(numFeatures, numClasses)
        val lm = LinearMapper[DenseVector[Double]](weights, Some(labelScaler.mean), Some(featureScaler))
        val testAcc = epochCallback.get(lm)
        println(s"EPOCH_${epoch}_LAMBDA_${regParam}_TEST_ACC_${testAcc}")
        //println("For epoch " + epoch + " TEST accuracy " + epochCallback.get(lm))
      }

			epoch = epoch + 1
    }
    val finalWeights = currentWeights.asDenseMatrix.reshape(numFeatures, numClasses)

    val lossHistoryArray = lossHistory.result()

    logInfo("MiniBatchSGDwithL2.runSGD finished. Last 10 losses %s".format(
      lossHistoryArray.takeRight(10).mkString(", ")))

    finalWeights
  }

  /**
   * CostFun implements Breeze's DiffFunction[T], which returns the loss and gradient
   * at a particular point (weights). It's used in Breeze's convex optimization routines.
   */
  private class GradFun(
    dataMat: RDD[DenseMatrix[Double]],
		dataColMeans: DenseVector[Double],
    dataColStdevs: Option[DenseVector[Double]],
    labelsMat: RDD[DenseMatrix[Double]],
    gradient: BatchGradient,
    regParam: Double,
    miniBatchFraction: Double,
    numExamples: Long,
    numFeatures: Int,
    numClasses: Int) {
    def calculate(weights: DenseVector[Double]): (Double, DenseVector[Double]) = {
      val weightsMat = weights.asDenseMatrix.reshape(numFeatures, numClasses)
      // Have a local copy to avoid the serialization of CostFun object which is not serializable.
      val bcW = dataMat.context.broadcast(weightsMat)
			val localColMeansBC = dataMat.context.broadcast(dataColMeans)
      val localColStdevsBC = dataMat.context.broadcast(dataColStdevs)
      val localGradient = gradient

      val (gradientSum, lossSum) = MLMatrixUtils.treeReduce(dataMat.zip(labelsMat).map { x =>
          localGradient.compute(x._1, localColMeansBC.value, localColStdevsBC.value, x._2,
            bcW.value, miniBatchFraction)
        }, (a: (DenseMatrix[Double], Double), b: (DenseMatrix[Double], Double)) => { 
          a._1 += b._1
          (a._1, a._2 + b._2)
        }
      )

      // total loss = lossSum / nTrain + 1/2 * lambda * norm(W)^2
      val normWSquared = math.pow(norm(weights), 2)
      val regVal = 0.5 * regParam * normWSquared
      val loss = lossSum / numExamples + regVal

      // total gradient = gradSum / nTrain + lambda * w
      val gradientTotal = gradientSum / numExamples.toDouble + (weightsMat * regParam)

      (loss, gradientTotal.toDenseVector)
    }
  }

  def isConverged(
      previousWeights: DenseVector[Double],
      currentWeights: DenseVector[Double],
      convergenceTol: Double): Boolean = {

    if (previousWeights != null) {
      // This represents the difference of updated weights in the iteration.
      val solutionVecDiff: Double = norm(previousWeights - currentWeights)
      solutionVecDiff < convergenceTol * Math.max(norm(currentWeights), 1.0)
    } else {
      false
    } 
  }

}
