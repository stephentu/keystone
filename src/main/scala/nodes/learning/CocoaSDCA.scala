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
 * @param gradient Gradient function to be used.
 * @param convergenceTol convergence tolerance for L-BFGS
 * @param regParam L2 regularization
 * @param numIterations max number of iterations to run 
 *
 * @param localIters - number of inner localSDCA iterations (H in the paper)
 * @param beta - scaling parameter for CoCoA
 * @param gamma - aggregation parameter for CoCoA+ (gamma=1 for adding, gamma=1/K for averaging) 
 */
class CocoaSDCAwithL2(
    val gradient: BatchGradient, 
    val convergenceTol: Double = 1e-4,
    val numIterations: Int = 100,
    val regParam: Double = 0.0,
    val normStd: Boolean = false,
    val numLocalItersFraction: Double = 1.0,
    val gamma: Double = 1.0,
    val beta: Double = 1.0,
    val computeCost: Boolean = false,
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
    val popFeatureMean = MatrixUtils.computeColMean(data, numExamples, numFeatures)
    val popStdEv = if (normStd) {
      Some(MatrixUtils.computeColStdEv(data, numExamples, popFeatureMean, numFeatures))
    } else {
      None
    }
		val featureScaler = new StandardScalerModel(popFeatureMean, popStdEv)
		val labelScaler = new StandardScalerModel(MatrixUtils.computeColMean(labels, numExamples, numClasses), None)
    // val featureScaler = new StandardScaler(normalizeStdDev = false).fit(data)
    // val labelScaler = new StandardScaler(normalizeStdDev = false).fit(labels)

    val model = CocoaSDCAwithL2.runCocoa(
      data,
      labels,
      featureScaler,
      labelScaler,
      gradient,
      convergenceTol,
      numIterations,
      regParam,
      numLocalItersFraction,
      gamma,
      beta,
      computeCost,
      epochCallback,
      epochEveryTest)

    (model, Some(labelScaler.mean), Some(featureScaler))
  }

}

object CocoaSDCAwithL2 extends Logging {

  def runCocoa(
      data: RDD[DenseMatrix[Double]],
      labels: RDD[DenseMatrix[Double]],
      featureScaler: StandardScalerModel,
      labelScaler: StandardScalerModel,
      gradient: BatchGradient,
      convergenceTol: Double,
      maxNumIterations: Int,
      regParam: Double,
      numLocalItersFraction: Double,
      gamma: Double,
      beta: Double,
      computeCost: Boolean,
      epochCallback: Option[LinearMapper[DenseVector[Double]] => Double] = None,
      epochEveryTest: Int = 10): DenseMatrix[Double] = {

    // TODO: Add this as a parameter ?!
    val plus = false
    val chkptIter = 20
    val seed = 12654764

    val parts = data.partitions.size
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

    val dataAndLabels = data.zip(labelsMat)

    var alpha = labels.map(x => DenseMatrix.zeros[Double](x.rows, x.cols)).cache()
    val scaling = if (plus) gamma else beta/parts
    println("Scaling is " + scaling)

    val initialWeights = DenseMatrix.zeros[Double](numFeatures, numClasses)
    var prevWeights = null
    var currentWeights = initialWeights

    var epoch = 0
    while (epoch < maxNumIterations && !isConverged(prevWeights, currentWeights, convergenceTol)) {
      val epochBegin = System.nanoTime

      // zip alpha with data
      val zipData = alpha.zip(dataAndLabels)

      // find updates to alpha, w
      val updates = zipData.mapPartitions(
        partitionUpdate(_, currentWeights, numLocalItersFraction, regParam, numExamples, scaling,
          seed + epoch, plus, parts * gamma, featureScaler.mean, featureScaler.std), preservesPartitioning = true).persist()
      alpha = updates.map(kv => kv._2)
      val primalUpdates = updates.map(kv => kv._1).reduce(_ + _)
      currentWeights += (primalUpdates * scaling)

      // optionally checkpoint RDDs
      if(!data.context.getCheckpointDir.isEmpty && epoch % chkptIter == 0) {
        zipData.checkpoint()
        alpha.checkpoint()
      }
      val prevWeights = currentWeights
      val epochTime = System.nanoTime - epochBegin

      println("For epoch " + epoch + " x norm " + norm(prevWeights.toDenseVector))
      if (computeCost) {
        val cost = LinearMapEstimator.computeCost(featureScaler.apply(data.flatMap(x =>
          MatrixUtils.matrixToRowArray(x).iterator)), labelsMat.flatMap(x =>
          MatrixUtils.matrixToRowArray(x).iterator), regParam, currentWeights, None)
        println("For epoch " + epoch + " cost " + cost)
      }
      println("EPOCH_" + epoch + "_time: " + epochTime)
      if (!epochCallback.isEmpty && epoch % epochEveryTest == 1) {
        val lm = LinearMapper[DenseVector[Double]](currentWeights, Some(labelScaler.mean), Some(featureScaler))
        val testAcc = epochCallback.get(lm)
        println(s"EPOCH_${epoch}_LAMBDA_${regParam}_TEST_ACC_${testAcc}")
        //println("For epoch " + epoch + " TEST accuracy " + epochCallback.get(lm))
      }

			epoch = epoch + 1
    }
    logInfo("CocoaSDCAwithL2.runCocoa finished")

    currentWeights
  }

  def isConverged(
      previousWeights: DenseMatrix[Double],
      currentWeights: DenseMatrix[Double],
      convergenceTol: Double): Boolean = {

    if (previousWeights != null) {
      // This represents the difference of updated weights in the iteration.
      val solutionVecDiff: Double = norm((previousWeights - currentWeights).toDenseVector)
      solutionVecDiff < convergenceTol * Math.max(norm(currentWeights.toDenseVector), 1.0)
    } else {
      false
    } 
  }

  /**
   * Performs one round of local updates using a given local dual algorithm, 
   * here locaSDCA. Will perform localIters many updates per worker.
   *
   * @param zipData
   * @param winit
   * @param localIters
   * @param lambda
   * @param n
   * @param scaling This is either gamma for CoCoA+ or beta/K for CoCoA
   * @param seed
   * @param plus
   * @param sigma sigma' in the CoCoA+ paper
   * @return
   */
  private def partitionUpdate(
    zipData: Iterator[(DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double]))],
    wInit: DenseMatrix[Double], 
    localItersFraction: Double, 
    lambda: Double, 
    n: Int, 
    scaling: Double,
    seed: Int,
    plus: Boolean,
    sigma: Double,
    featureMeans: DenseVector[Double],
    featureStdEv: Option[DenseVector[Double]]): Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = {

    val zipPair = zipData.next()
    val localFeatures = zipPair._2._1
    val localLabels = zipPair._2._2
    var alpha = zipPair._1
    val alphaOld = alpha.copy

    val (deltaAlpha, deltaW) = localSDCA(
      localFeatures, localLabels, wInit, localItersFraction, lambda, n, alpha,
      alphaOld, seed, plus, sigma, featureMeans, featureStdEv)
    alpha = alphaOld + (deltaAlpha * scaling)

    return Iterator.single((deltaW, alpha))
  }

  /**
   * This is an implementation of LocalDualMethod, here LocalSDCA (coordinate ascent),
   * with taking the information of the other workers into account, by respecting the
   * shared wInit vector.
   * Here we perform coordinate updates for the SVM dual objective (hinge loss).
   *
   * Note that SDCA for hinge-loss is equivalent to LibLinear, where using the
   * regularization parameter  C = 1.0/(lambda*numExamples), and re-scaling
   * the alpha variables with 1/C.
   *
   * @param localData The local data examples
   * @param wInit
   * @param localIters Number of local coordinates to update
   * @param lambda
   * @param n Global number of points (needed for the primal-dual correspondence)
   * @param alpha
   * @param alphaOld
   * @param seed
   * @param plus
   * @param sigma
   * @param plus
   * @return (deltaAlpha, deltaW) Summarizing the performed local changes
   */
  def localSDCA(
    localFeatures: DenseMatrix[Double],
    localLabels: DenseMatrix[Double],
    wInit: DenseMatrix[Double],
    localItersFraction: Double,
    lambda: Double, 
    n: Int,
    alpha: DenseMatrix[Double], 
    alphaOld: DenseMatrix[Double],
    seed: Int,
    plus: Boolean,
    sigma: Double,
    featureMeans: DenseVector[Double],
    featureStdEv: Option[DenseVector[Double]]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    
    var w = wInit
    val nLocal = localFeatures.rows
    val nD = n.toDouble
    var r = new scala.util.Random(seed)
    var deltaW = DenseMatrix.zeros[Double](wInit.rows, wInit.cols)

    val localIters = math.ceil(nLocal * localItersFraction).toInt

    // perform local udpates
    for (i <- 1 to localIters) {

      // randomly select a local example
      val idx = r.nextInt(nLocal)
      val x = localFeatures(idx, ::).t - featureMeans
      featureStdEv.foreach(stdev => x :/= stdev)
      val y = localLabels(idx, ::).t

      // compute hinge loss gradient
      // val grad = {
      //   if (plus) {
      //     (y * (x.dot(w) + (sigma * x.dot(deltaW))) - 1.0) * (lambda * n)
      //   } else {
      //     (y * (x.dot(w)) - 1.0) * (lambda * n)
      //   }
      // }
      // // compute projected gradient
      // var proj_grad = grad
      // if (alpha(idx) <= 0.0)
      //   proj_grad = Math.min(grad, 0)
      // else if (alpha(idx) >= 1.0)
      //   proj_grad = Math.max(grad, 0)

      // compute square loss gradient
      val del_alpha = if (!plus) {
        (y - w.t * x - nD * alpha(idx, ::).t) / (nD + norm(x)/lambda)
      } else {
        (y - (w.t * x) * sigma - nD * alpha(idx, ::).t) / (nD + (norm(x) * sigma) / lambda)
      }
      val newAlpha = alpha(idx, ::) + del_alpha.t

      // update primal and dual variables
      val update = (x * del_alpha.t) * (1.0/lambda)
      // println("update norm " + norm(update.toDenseVector))
      if (!plus) {
        w += update
      }
      deltaW += update
      alpha(idx, ::) := newAlpha
    }

    val deltaAlpha = alpha - alphaOld

    return (deltaAlpha, deltaW)
  }

}
