package nodes.learning

import breeze.linalg._
import breeze.numerics._

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import evaluation.{MulticlassClassifierEvaluator, MulticlassMetrics}
import loaders.LabeledData
import nodes.stats.GaussianKernel
import nodes.util._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.SparkContext._
import pipelines._
import utils.{Image, MatrixUtils}
import workflow.Pipeline

import edu.berkeley.cs.amplab.mlmatrix.util.{Utils => MLMatrixUtils}

import org.apache.spark.rdd.RDD


/**
 * TODO: this is very un-keystoneish for now
 */

case class DCSolverState(
    gamma: Double,
    kmeans: KMeansModel,
    models: RDD[(Int, (DenseMatrix[Double], DenseMatrix[Double]))], /* (partId, (Xtrain_part, alphaStar_part)) */
    trainEval: MulticlassMetrics) extends Logging {

  def metrics(test: LabeledData[Int, DenseVector[Double]],
              numClasses: Int /* should not need to pass this around */):
    MulticlassMetrics = {

    val testData = models.partitioner.map { partitioner =>
      kmeans(test.data).map(DCSolver.oneHotToNumber).zip(test.labeledData).groupByKey(partitioner)
    }.getOrElse {
      logWarning("Could not find partitioner-- join could be slow")
      kmeans(test.data).map(DCSolver.oneHotToNumber).zip(test.labeledData).groupByKey()
    }

    val testEvaluationStartTime = System.nanoTime()
    val testEvaluation = models.join(testData).mapValues { case ((xtrain, alphaStar), rhs) =>
      val rhsSeq = rhs.toSeq
      val Xtest = MatrixUtils.rowsToMatrix(rhsSeq.map(_._2))
      val Ytest = rhsSeq.map(_._1)
      val Ktesttrain = GaussianKernel(gamma).apply(Xtest, xtrain)
      val testPredictions = MatrixUtils.matrixToRowArray(Ktesttrain * alphaStar).map(MaxClassifier.apply)
      (Ytest, testPredictions)
    }.cache()
    testEvaluation.count()
    logInfo(s"Test evaluation took ${(System.nanoTime() - testEvaluationStartTime)/1e9} s")

    val flattenedTestLabels = testEvaluation.map(_._2._1).flatMap(x => x)
    val flattenedTestPredictions = testEvaluation.map(_._2._2).flatMap(x => x)

    MulticlassClassifierEvaluator(flattenedTestPredictions, flattenedTestLabels, numClasses)
  }

}

case class DCSolverYuchenState(
  gamma: Double,
  models: RDD[(Int, (DenseMatrix[Double], DenseMatrix[Double]))] /* it is assumed each entry in the RDD belongs to a partition */
) extends Logging {


  def metrics(test: LabeledData[Int, DenseVector[Double]],
              numClasses: Int): MulticlassMetrics = {

    val nModels = models.count()
    assert(nModels == models.partitions.size)

    var testAccums = test.labeledData.mapPartitions { partition =>
      Iterator.single(DenseMatrix.zeros[Double](partition.size, numClasses))
    }.cache()
    testAccums.count()

    (0 until nModels.toInt).foreach { modelId =>

      // not sure of a better way to do this
      val (_, model) = models.filter { case (curModelId, _) => curModelId == modelId }.collect().head
      val modelBC = test.labeledData.context.broadcast(model)

      val newTestAccums = testAccums.zipPartitions(test.labeledData) { case (accums, partition) =>
        val accum = accums.next()
        val partitionSeq = partition.toSeq

        val Xtest = MatrixUtils.rowsToMatrix(partitionSeq.map(_._2)) // DenseMatrix[Double]
        val Ytest = partitionSeq.map(_._1) // Seq[Int]

        val (xTrainPart, alphaStarPart) = modelBC.value
        val KtesttrainPart = GaussianKernel(gamma).apply(Xtest, xTrainPart)

        accum += KtesttrainPart * alphaStarPart
        Iterator.single(accum)
      }.cache()
      newTestAccums.count()
      testAccums.unpersist(true)
      testAccums = newTestAccums

      modelBC.destroy()
      // TODO: truncate this lineage?
    }

    MulticlassClassifierEvaluator(
      testAccums.map { evaluations =>
        MatrixUtils.matrixToRowArray(evaluations * (1.0 / nModels.toDouble)).map(MaxClassifier.apply).toSeq
      }.flatMap(x => x),
      test.labels,
      numClasses)
  }
}

object DCSolverYuchen extends Logging {

  def fit(train: LabeledData[Int, DenseVector[Double]],
          numClasses: Int,
          lambda: Double,
          gamma: Double,
          numPartitions: Int,
          permutationSeed: Long): DCSolverYuchenState = {

    val sc = train.labeledData.context

    // TODO: FIXME
    val shuffledTrain = if (false) {
      val rng = new util.Random(permutationSeed)
      val pi = sc.parallelize(rng.shuffle((0 until train.labeledData.count().toInt).toIndexedSeq), numPartitions)
      // TODO: we might want to not persist this stage so we don't have to materialize 2X of the train data
      val shuffledTrain = pi.zip(train.labeledData).sortByKey().map(_._2).repartition(numPartitions).cache()
      shuffledTrain.count()
      shuffledTrain
    } else {
      train.labeledData
    }

    val models = shuffledTrain.mapPartitionsWithIndex { case (partId, partition) =>
      val partitionSeq = partition.toSeq
      val classLabeler = ClassLabelIndicatorsFromIntLabels(numClasses)
      val Xtrain = MatrixUtils.rowsToMatrix(partitionSeq.map(_._2))
      val Ytrain = MatrixUtils.rowsToMatrix(partitionSeq.map(_._1).map(classLabeler.apply))

      val localKernelStartTime = System.nanoTime()
      val Ktrain = GaussianKernel(gamma).apply(Xtrain)
      logInfo(s"[${partId}] Local kernel gen took ${(System.nanoTime() - localKernelStartTime)/1e9} s")

      val localSolveStartTime = System.nanoTime()
      val alphaStar = (Ktrain + (DenseMatrix.eye[Double](Ktrain.rows) :* lambda)) \ Ytrain
      logInfo(s"[${partId}] Local solve took ${(System.nanoTime() - localSolveStartTime)/1e9} s")

      Iterator.single((partId, (Xtrain, alphaStar)))
    }.cache()
    models.count()
    shuffledTrain.unpersist()

    DCSolverYuchenState(gamma, models)
  }

}

/**
 * L2 kernel RR is hard-coded for now
 */
object DCSolver extends Logging {

  // necessary since kmeans returns one-hot encoded vectors
  def oneHotToNumber(x: DenseVector[Double]): Int = {
    var i = 0
    while (i < x.size) {
      if (x(i) == 1.0)
        return i
      i += 1
    }
    throw new RuntimeException("should not get here")
  }

  def fit(train: LabeledData[Int, DenseVector[Double]],
          numClasses: Int, /* should not need to pass this around */
          lambda: Double,
          gamma: Double,
          numPartitions: Int,
          kmeansSampleSize: Double,
          kmeansSeed: Long): DCSolverState = {

    val kmeansStartTime = System.nanoTime()
    val kmeans = {
      val trainSubsample = train.data.sample(false, kmeansSampleSize, kmeansSeed)
      KMeansPlusPlusEstimator(numPartitions, 100).fit(trainSubsample)
    }
    logInfo(s"KMeans took ${(System.nanoTime() - kmeansStartTime)/1e9} s")

    val trainingStartTime = System.nanoTime()
    val models = kmeans(train.data)
      .map(oneHotToNumber)
      .zip(train.labeledData)
      .groupByKey()
      .map { case (partId, partition) =>
          val elems = partition.toSeq
          val classLabeler = ClassLabelIndicatorsFromIntLabels(numClasses)
          val Xtrain = MatrixUtils.rowsToMatrix(elems.map(_._2))
          val Ytrain = MatrixUtils.rowsToMatrix(elems.map(_._1).map(classLabeler.apply))

          val localKernelStartTime = System.nanoTime()
          val Ktrain = GaussianKernel(gamma).apply(Xtrain)
          logInfo(s"[${partId}] Local kernel gen took ${(System.nanoTime() - localKernelStartTime)/1e9} s")

          val localSolveStartTime = System.nanoTime()
          val alphaStar = (Ktrain + (DenseMatrix.eye[Double](Ktrain.rows) :* lambda)) \ Ytrain
          logInfo(s"[${partId}] Local solve took ${(System.nanoTime() - localSolveStartTime)/1e9} s")

          // evaluate training error-- we can do this now for DC-SVM since
          // the model used for prediction is the one associated with the center
          val predictions = MatrixUtils.matrixToRowArray(Ktrain * alphaStar).map(MaxClassifier.apply).toSeq
          val trainLabels = elems.map(_._1)
          assert(predictions.size == trainLabels.size)

          val nErrors = predictions.zip(trainLabels).filter { case (x, y) => x != y }.size
          val trainErrorRate = (nErrors.toDouble / Xtrain.rows.toDouble)
          val trainAccRate = 1.0 - trainErrorRate

          logInfo(s"[${partId}] localTrainEval acc: ${trainAccRate}, err: ${trainErrorRate}, nErrors: ${nErrors}, nLocal: ${Xtrain.rows}")

          (partId, (Xtrain, alphaStar, trainLabels, predictions))
        }.cache()
    models.count()
    logInfo(s"Training took ${(System.nanoTime() - trainingStartTime)/1e9} s")

    val flattenedTrainLabels = models.map(_._2._3).flatMap(x => x)
    val flattenedTrainPredictions = models.map(_._2._4).flatMap(x => x)

    val trainEval = MulticlassClassifierEvaluator(flattenedTrainPredictions, flattenedTrainLabels, numClasses)
    DCSolverState(gamma, kmeans, models.mapValues { case (x, w, _, _) => (x, w) }, trainEval)
  }

}



