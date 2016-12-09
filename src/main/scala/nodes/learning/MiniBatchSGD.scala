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
    val dampen: Option[Double] = None,
    val miniBatchFraction: Double = 0.1,
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
    val featureScaler = new StandardScalerModel(popFeatureMean, None)
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
      dampen,
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

  /**
   * Run Stochastic Gradient Descent in parallel.
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
      dampen: Option[Double],
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

    val gradFun = new GradFun(data, featureScaler.mean, labelsMat, gradient, regParam,
      miniBatchFraction, numExamples, numFeatures, numClasses)

    println("miniBatchFraction: " + miniBatchFraction + ", dampen: " + dampen)
    println("MAX ROW NORM IS " + gradFun.maxRowNorm())

    val initialWeights = DenseVector.zeros[Double](numFeatures * numClasses)

    val itersPerEpoch = Math.ceil(1.0 / miniBatchFraction).toInt
    println("itersPerEpoch: " + itersPerEpoch)

    /**
     * NOTE: lossSum and loss is computed using the weights from the previous iteration
     * and regVal is the regularization value computed in the previous iteration as well.
     */
    var prevWeights = null
    var currentWeights = initialWeights
    var currentLoss = 0.0
    var iter = 1
    var thisIterStepSize = stepSize
    while (iter <= maxNumIterations && !isConverged(prevWeights, currentWeights, convergenceTol)) {
      val iterBegin = System.nanoTime
      val (loss, gradient) = gradFun.calculate(currentWeights)

      lossHistory += loss
      val prevWeights = currentWeights

      // NOTE: Since we included the regularization term in the computation of
      // gradient, we only need to do this here.
      currentWeights = prevWeights - thisIterStepSize * gradient

      println("For iter " + iter + " step size " + thisIterStepSize)
      println("For iter " + iter + " loss is " + loss)
      println("For iter " + iter + " grad norm is " + norm(gradient))
      println("For iter " + iter + " x norm " + norm(prevWeights))
      println("For iter " + iter + " new x norm " + norm(currentWeights))
      val iterTime = System.nanoTime - iterBegin
      println("iter_" + iter + "_time: " + iterTime)
      if (!epochCallback.isEmpty && (epochEveryTest == 1 || iter % epochEveryTest == 1)) {
        val weights = currentWeights.asDenseMatrix.reshape(numFeatures, numClasses)
        val lm = LinearMapper[DenseVector[Double]](weights, Some(labelScaler.mean), Some(featureScaler))
        val testAcc = epochCallback.get(lm)
        println(s"ITER_${iter}_LAMBDA_${regParam}_TEST_ACC_${testAcc}")
        //println("For iter " + iter + " TEST accuracy " + epochCallback.get(lm))
      }

      if ((iter % itersPerEpoch) == 0) {
        dampen.foreach { rho =>
          thisIterStepSize *= rho
        }
      }

      iter = iter + 1
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
    labelsMat: RDD[DenseMatrix[Double]],
    gradient: BatchGradient,
    regParam: Double,
    miniBatchFraction: Double,
    numExamples: Long,
    numFeatures: Int,
    numClasses: Int) {

    def maxRowNorm(): Double = {
      val localColMeansBC = dataMat.context.broadcast(dataColMeans)
      val rowNorms = dataMat.map { x =>
        var i = 0
        var max_row_norm = 0.0
        while (i < x.rows) {
          val x_zm = x(i, ::).t - localColMeansBC.value
          max_row_norm = math.max(max_row_norm, norm(x_zm))
          i = i + 1
        }
        max_row_norm
      }

      rowNorms.collect().max
    }

    def calculate(weights: DenseVector[Double]): (Double, DenseVector[Double]) = {
      val weightsMat = weights.asDenseMatrix.reshape(numFeatures, numClasses)
      // Have a local copy to avoid the serialization of CostFun object which is not serializable.
      val bcW = dataMat.context.broadcast(weightsMat)
      val localColMeansBC = dataMat.context.broadcast(dataColMeans)
      val localGradient = gradient
      val localMiniBatchFraction = miniBatchFraction
      
      val (gradientSum, lossSum) = MLMatrixUtils.treeReduce(dataMat.zip(labelsMat).map { x =>
          localGradient.compute(x._1, localColMeansBC.value, x._2,
            bcW.value, localMiniBatchFraction)
        }, (a: (DenseMatrix[Double], Double), b: (DenseMatrix[Double], Double)) => {
          a._1 += b._1
          (a._1, a._2 + b._2)
        }
      )
    

      // total loss = lossSum / nTrain + 1/2 * lambda * norm(W)^2
      val normWSquared = math.pow(norm(weights), 2)
      val regVal = 0.5 * regParam * normWSquared
      val loss = lossSum / math.ceil(numExamples * miniBatchFraction) + regVal

      localColMeansBC.destroy()
      bcW.destroy()

      // total gradient = gradSum / nTrain + lambda * w
      val gradientTotal = gradientSum / math.ceil(numExamples * miniBatchFraction) + (weightsMat * regParam)

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
