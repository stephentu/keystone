package nodes.learning

import breeze.linalg._
import breeze.numerics._

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import evaluation.{AugmentedExamplesEvaluator, MulticlassClassifierEvaluator, MulticlassMetrics}
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
    lambdas: Seq[Double],
    gamma: Double,
    kmeans: KMeansModel,
    models: RDD[(Int, (DenseMatrix[Double], Seq[DenseMatrix[Double]]))], /* (partId, (Xtrain_part, Seq[alphaStar_part])) */
    trainEvals: Seq[MulticlassMetrics]) extends Logging {

  def metrics(test: LabeledData[Int, DenseVector[Double]],
              numClasses: Int /* should not need to pass this around */):
    Seq[MulticlassMetrics] = {

    val testData = models.partitioner.map { partitioner =>
      kmeans(test.data).map(DCSolver.oneHotToNumber).zip(test.labeledData).groupByKey(partitioner)
    }.getOrElse {
      logWarning("Could not find partitioner-- join could be slow")
      kmeans(test.data).map(DCSolver.oneHotToNumber).zip(test.labeledData).groupByKey()
    }

    val testEvaluationStartTime = System.nanoTime()
    val testEvaluation = models.join(testData).mapValues { case ((xtrain, alphaStars), rhs) =>
      val rhsSeq = rhs.toSeq
      val Xtest = MatrixUtils.rowsToMatrix(rhsSeq.map(_._2))
      val Ytest = rhsSeq.map(_._1)
      val Ktesttrain = GaussianKernel(gamma).apply(Xtest, xtrain)
      val allTestPredictions = alphaStars.map { alphaStar =>
        MatrixUtils.matrixToRowArray(Ktesttrain * alphaStar).map(MaxClassifier.apply)
      }
      (Ytest, allTestPredictions)
    }.cache()
    testEvaluation.count()
    val totalTestTime = (System.nanoTime() - testEvaluationStartTime)/1e9
    println(s"TEST_TIME_${totalTestTime}")

    val flattenedTestLabels = testEvaluation.map(_._2._1).flatMap(x => x)
    (0 until lambdas.size).map { idx =>
      val flattenedTestPredictions = testEvaluation.map(_._2._2.apply(idx)).flatMap(x => x)
      MulticlassClassifierEvaluator(flattenedTestPredictions, flattenedTestLabels, numClasses)
    }
  }

  def augmentedMetrics(
      test: RDD[(String, Int, DenseVector[Double])],
      numClasses: Int /* should not need to pass this around */): Seq[MulticlassMetrics] = {

    val testData = models.partitioner.map { partitioner =>
      kmeans(test.map(_._3)).map(DCSolver.oneHotToNumber).zip(test).groupByKey(partitioner)
    }.getOrElse {
      logWarning("Could not find partitioner-- join could be slow")
      kmeans(test.map(_._3)).map(DCSolver.oneHotToNumber).zip(test).groupByKey()
    }

    val testEvaluationStartTime = System.nanoTime()
    val testEvaluation = models.join(testData).mapValues { case ((xtrain, alphaStars), rhs) =>
      val rhsSeq = rhs.toSeq
      // every element in rhs is ((Int, DV[Double]), String)
      val Xtest = MatrixUtils.rowsToMatrix(rhsSeq.map(_._3))
      val Ytest = rhsSeq.map(_._2)
      val Ktesttrain = GaussianKernel(gamma).apply(Xtest, xtrain)
      val allTestPredictions = alphaStars.map { alphaStar =>
        MatrixUtils.matrixToRowArray(Ktesttrain * alphaStar)
      }
      (Ytest, allTestPredictions, rhsSeq.map(_._1))
    }.cache()
    testEvaluation.count()
    val totalTestTime = (System.nanoTime() - testEvaluationStartTime)/1e9
    println(s"TEST_TIME_${totalTestTime}")
    //logInfo(s"Test evaluation took ${(System.nanoTime() - testEvaluationStartTime)/1e9} s")

    val flattenedTestLabels = testEvaluation.map(_._2._1).flatMap(x => x)
    val flattenedImageIds = testEvaluation.map(_._2._3).flatMap(x => x)
    val result = (0 until lambdas.size).map { idx =>
      val flattenedTestPredictions = testEvaluation.map(_._2._2.apply(idx)).flatMap(x => x)
      AugmentedExamplesEvaluator(flattenedImageIds, flattenedTestPredictions, flattenedTestLabels, numClasses)
    }
    logInfo("Done augmented example evaluation")
    result
  }



}

case class DCSolverYuchenState(
  lambdas: Seq[Double],
  gamma: Double,
  models: RDD[(Int, (DenseMatrix[Double], Seq[DenseMatrix[Double]]))] /* it is assumed each entry in the RDD belongs to a partition */
) extends Logging {


  def metrics(test: LabeledData[Int, DenseVector[Double]],
              numClasses: Int): Seq[MulticlassMetrics] = {

    val nModels = models.count()
    assert(nModels == models.partitions.size)

    var testAccums = test.labeledData.mapPartitions { partition =>
      val sz = partition.size
      assert(sz > 0)
      Iterator.single(lambdas.map { _ => DenseMatrix.zeros[Double](sz, numClasses) })
    }.cache()
    testAccums.count()

    (0 until nModels.toInt).foreach { modelId =>

      // not sure of a better way to do this
      val (_, model) = models.filter { case (curModelId, _) => curModelId == modelId }.collect().head
      val modelBC = test.labeledData.context.broadcast(model)

      val newTestAccums = testAccums.zipPartitions(test.labeledData) { case (accumsIter, partition) =>
        val accums = accumsIter.next()
        val partitionSeq = partition.toSeq

        val Xtest = MatrixUtils.rowsToMatrix(partitionSeq.map(_._2)) // DenseMatrix[Double]
        val Ytest = partitionSeq.map(_._1) // Seq[Int]

        val (xTrainPart, alphaStarsPart) = modelBC.value
        val KtesttrainPart = GaussianKernel(gamma).apply(Xtest, xTrainPart)

        accums.zip(alphaStarsPart).foreach { case (accum, alphaStarPart) =>
          accum += KtesttrainPart * alphaStarPart
        }

        Iterator.single(accums)
      }.cache()
      newTestAccums.count()
      testAccums.unpersist()
      testAccums = newTestAccums

      //modelBC.destroy() // LEAK FOR NOW
      modelBC.unpersist()
      // TODO: truncate this lineage?
    }

    (0 until lambdas.size).map { idx =>
      MulticlassClassifierEvaluator(
        testAccums.map { evaluations: Seq[DenseMatrix[Double]] =>
          assert(lambdas.size == evaluations.size)
          val thisEvaluations = evaluations(idx)
          assert(thisEvaluations.cols == numClasses)
          MatrixUtils.matrixToRowArray(evaluations(idx) * (1.0 / nModels.toDouble)).map(MaxClassifier.apply).toSeq
        }.flatMap(x => x),
        test.labels,
        numClasses)
    }
  }
  def augmentedMetrics(
      test: RDD[(String, Int, DenseVector[Double])],
      numClasses: Int /* should not need to pass this around */): Seq[MulticlassMetrics] = {

    //DEBUG: print number of elements in partition
    val testNum = test.count
    val testNumPerPartition = test.mapPartitions { iter => Iterator.single(iter.size) }.collect()
    println("testNum: " + testNum)
    println("test per partition: " + testNumPerPartition.mkString(","))

    val nModels = models.count()
    assert(nModels == models.partitions.size)

    var testAccums = test.mapPartitions { partition =>
      if (partition.isEmpty) {
        Iterator.empty
      } else {
        val sz = partition.size
        assert(sz > 0)
        Iterator.single(lambdas.map { _ => DenseMatrix.zeros[Double](sz, numClasses) })
    }}.cache()
    testAccums.count()

    (0 until nModels.toInt).foreach { modelId =>

      // not sure of a better way to do this
      val (_, model) = models.filter { case (curModelId, _) => curModelId == modelId }.collect().head
      val modelBC = test.context.broadcast(model)

      val newTestAccums = testAccums.zipPartitions(test) { case (accumsIter, partition) =>
        if (!accumsIter.isEmpty) {
          val accums = accumsIter.next()
          val partitionSeq = partition.toSeq

          val Xtest = MatrixUtils.rowsToMatrix(partitionSeq.map(_._3)) // DenseMatrix[Double]
          val Ytest = partitionSeq.map(_._2) // Seq[Int]

          val (xTrainPart, alphaStarsPart) = modelBC.value
          val KtesttrainPart = GaussianKernel(gamma).apply(Xtest, xTrainPart)

          accums.zip(alphaStarsPart).foreach { case (accum, alphaStarPart) =>
            accum += KtesttrainPart * alphaStarPart
          }

          Iterator.single(accums)
        } else {
          Iterator.empty
        }
      }.cache()
      newTestAccums.count()
      testAccums.unpersist()
      testAccums = newTestAccums

      //modelBC.destroy() // LEAK FOR NOW
      modelBC.unpersist()
      // TODO: truncate this lineage?
    }

    val testLabels = test.map(_._2)
    val imageIds = test.map(_._1)
    (0 until lambdas.size).map { idx =>
      val testPredictions = testAccums.map { evaluations =>
        assert(lambdas.size == evaluations.size)
        val thisEvaluations = evaluations(idx)
        assert(thisEvaluations.cols == numClasses)
        MatrixUtils.matrixToRowArray(evaluations(idx) * (1.0 / nModels.toDouble))
      }.flatMap( x => x)
      AugmentedExamplesEvaluator(imageIds, testPredictions, testLabels, numClasses)
    }
  }
}

object DCSolverYuchen extends Logging {

  private def addLambdaEyeInPlace(in: DenseMatrix[Double], lambda: Double): Unit = {
    assert(in.rows == in.cols)
    var i = 0
    while (i < in.rows) {
      in(i, i) += lambda
      i += 1
    }
  }

  private def subLambdaEyeInPlace(in: DenseMatrix[Double], lambda: Double): Unit = {
    assert(in.rows == in.cols)
    var i = 0
    while (i < in.rows) {
      in(i, i) -= lambda
      i += 1
    }
  }

  def fit(train: LabeledData[Int, DenseVector[Double]],
          numClasses: Int,
          lambdas: Seq[Double],
          gamma: Double,
          numPartitions: Int,
          permutationSeed: Long): DCSolverYuchenState = {

    val sc = train.labeledData.context

    val rng = new util.Random(permutationSeed)
    val pi: Array[Int] = rng.shuffle((0 until train.labeledData.count().toInt).toIndexedSeq).toArray
    val piBC = sc.broadcast(pi)

    val shuffledTrain = train.labeledData
      .zipWithIndex
      .map { case (elem, idx) => (piBC.value.apply(idx.toInt), elem) }
      .sortByKey(true, numPartitions)
      .map(_._2)
      .cache()
    shuffledTrain.count()

    piBC.destroy()

    val models = shuffledTrain.mapPartitionsWithIndex { case (partId, partition) =>
      val partitionSeq = partition.toSeq
      val classLabeler = ClassLabelIndicatorsFromIntLabels(numClasses)
      val Xtrain = MatrixUtils.rowsToMatrix(partitionSeq.map(_._2))
      val Ytrain = MatrixUtils.rowsToMatrix(partitionSeq.map(_._1).map(classLabeler.apply))

      val localKernelStartTime = System.nanoTime()
      val Ktrain = GaussianKernel(gamma).apply(Xtrain)
      logInfo(s"[${partId}] Local kernel gen took ${(System.nanoTime() - localKernelStartTime)/1e9} s")

      val alphaStars = lambdas.map { lambda =>
        val localSolveStartTime = System.nanoTime()
        addLambdaEyeInPlace(Ktrain, lambda)
        val alphaStar = Ktrain \ Ytrain
        logInfo(s"[${partId}] Local solve [lambda=${lambda}] took ${(System.nanoTime() - localSolveStartTime)/1e9} s")
        subLambdaEyeInPlace(Ktrain, lambda)

        val predictions = MatrixUtils.matrixToRowArray(Ktrain * alphaStar).map(MaxClassifier.apply).toSeq
        val trainLabels = partitionSeq.map(_._1)
        assert(predictions.size == trainLabels.size)

        val nErrors = predictions.zip(trainLabels).filter { case (x, y) => x != y }.size
        val trainErrorRate = (nErrors.toDouble / Xtrain.rows.toDouble)
        val trainAccRate = 1.0 - trainErrorRate

        println(s"PARTID_${partId}_LAMBDA_${lambda}_TRAIN_ACC_${trainAccRate}")
        //logInfo(s"[${partId}] localTrainEval lambda: ${lambda}, acc: ${trainAccRate}, err: ${trainErrorRate}, nErrors: ${nErrors}, nLocal: ${Xtrain.rows}")

        alphaStar
      }

      Iterator.single((partId, (Xtrain, alphaStars)))
    }.cache()
    models.count()
    shuffledTrain.unpersist()

    DCSolverYuchenState(lambdas, gamma, models)
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
          lambdas: Seq[Double],
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
      .groupByKey(numPartitions)
      .map { case (partId, partition) =>
          val elems = partition.toSeq
          val classLabeler = ClassLabelIndicatorsFromIntLabels(numClasses)
          val Xtrain = MatrixUtils.rowsToMatrix(elems.map(_._2))
          val Ytrain = MatrixUtils.rowsToMatrix(elems.map(_._1).map(classLabeler.apply))

          val localKernelStartTime = System.nanoTime()
          val Ktrain = GaussianKernel(gamma).apply(Xtrain)
          logInfo(s"[${partId}] Local kernel gen took ${(System.nanoTime() - localKernelStartTime)/1e9} s")

          val trainLabels = elems.map(_._1)

          val results = lambdas.map { lambda =>

            val localSolveStartTime = System.nanoTime()
            val alphaStar = (Ktrain + (DenseMatrix.eye[Double](Ktrain.rows) :* lambda)) \ Ytrain
            logInfo(s"[${partId}] Local solve took ${(System.nanoTime() - localSolveStartTime)/1e9} s")

            // evaluate training error-- we can do this now for DC-SVM since
            // the model used for prediction is the one associated with the center
            val predictions = MatrixUtils.matrixToRowArray(Ktrain * alphaStar).map(MaxClassifier.apply).toSeq
            assert(predictions.size == trainLabels.size)

            val nErrors = predictions.zip(trainLabels).filter { case (x, y) => x != y }.size
            val trainErrorRate = (nErrors.toDouble / Xtrain.rows.toDouble)
            val trainAccRate = 1.0 - trainErrorRate
            println(s"PARTID_${partId}_LAMBDA_${lambda}_TRAIN_ACC_${trainAccRate}")
            //logInfo(s"[${partId}] localTrainEval lambda: ${lambda}, acc: ${trainAccRate}, err: ${trainErrorRate}, nErrors: ${nErrors}, nLocal: ${Xtrain.rows}")

            (alphaStar, predictions)
          }

          (partId, (Xtrain, results.map(_._1), trainLabels, results.map(_._2)))
        }.cache()
    models.count()
    logInfo(s"Training took ${(System.nanoTime() - trainingStartTime)/1e9} s")

    val flattenedTrainLabels = models.map(_._2).map { case (_, _, labels, _) => labels }.flatMap(x => x)
    val trainEvals = (0 until lambdas.size).map { idx =>
      val flattenedTrainPredictions = models.map(_._2).flatMap { case (_, _, _, r) => r(idx) }
      MulticlassClassifierEvaluator(flattenedTrainPredictions, flattenedTrainLabels, numClasses)
    }

    DCSolverState(lambdas, gamma, kmeans, models.mapValues { case (x, as, _, _) => (x, as) }, trainEvals)
  }

}
