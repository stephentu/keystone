package pipelines.images.mnist

import breeze.linalg._
import breeze.numerics._

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import evaluation.MulticlassClassifierEvaluator
import loaders.{CsvDataLoader, LabeledData}
import nodes.learning.{KMeansModel, KMeansPlusPlusEstimator, BlockLeastSquaresEstimator}
import nodes.stats.{LinearRectifier, PaddedFFT, RandomSignNode, GaussianKernel}
import nodes.util._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import pipelines._
import scopt.OptionParser
import utils.{Image, MatrixUtils}
import workflow.Pipeline
import java.io.File


object MnistDCSolver extends Serializable with Logging {
  val appName = "MnistDCSolver"

  // http://stackoverflow.com/questions/1226555/case-class-to-map-in-scala
  private def ccAsMap(cc: AnyRef) =
    (Map[String, Any]() /: cc.getClass.getDeclaredFields) {(a, f) =>
      f.setAccessible(true)
      a + (f.getName -> f.get(cc))
    }

  private def froNormSquared(a: DenseMatrix[Double]): Double = {
    sum(a.values.map(x => math.pow(x, 2)))
  }

  def runSimpleSpark(sc: SparkContext, conf: MnistDCSolverConfig): Unit = {

    logInfo(ccAsMap(conf).toString)

    // This is a property of the MNIST Dataset (digits 0 - 9)
    val numClasses = 10

    val prng = new MersenneTwister(conf.seed)
    val kmeansSeed = prng.nextLong
    val randomSignSource = new RandBasis(new ThreadLocalRandomGenerator(prng))

    // The number of pixels in an MNIST image (28 x 28 = 784)
    // Because the mnistImageSize is 784, we get 512 PaddedFFT features per FFT.
    val mnistImageSize = 784

    val startTime = System.nanoTime()

    val train = LabeledData(
      CsvDataLoader(sc, conf.trainLocation, conf.numPartitions)
        // The pipeline expects 0-indexed class labels, but the labels in the file are 1-indexed
        .map(x => (x(0).toInt - 1, x(1 until x.length)))
        .map { case (y, x) => (y, (x * (2.0 / 255.0)) - 1.0) }
        .cache())

    val test = LabeledData(
      CsvDataLoader(sc, conf.testLocation, conf.numPartitions)
        // The pipeline expects 0-indexed class labels, but the labels in the file are 1-indexed
        .map(x => (x(0).toInt - 1, x(1 until x.length)))
        .map { case (y, x) => (y, (x * (2.0 / 255.0)) - 1.0) }
        .cache())

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

    val lambda = conf.lambda
    val gamma = conf.gamma

    val models = train.labeledData.zipWithIndex.map { case (datum, idx) =>
      (idx % conf.numPartitions, datum)
    }.groupByKey().map { case (partId, partition) =>
          val elems: Seq[(Int, DenseVector[Double])] = partition.toSeq
          println("[" + partId + "] elems.size: " + elems.size)
          val classLabeler = ClassLabelIndicatorsFromIntLabels(numClasses)
          val Xtrain = MatrixUtils.rowsToMatrix(elems.map(_._2))
          val Ytrain = MatrixUtils.rowsToMatrix(elems.map(_._1).map(classLabeler.apply))
          println("Xtrain.shape: " + Xtrain.rows + ", " + Xtrain.cols)
          println("Ytrain.shape: " + Ytrain.rows + ", " + Ytrain.cols)
          val Ktrain = GaussianKernel(gamma).apply(Xtrain)
          println("[" + partId + "] ||XTrain||_F " + froNormSquared(Xtrain) + ", ||KTrain||_F " + froNormSquared(Ktrain))
          assert(Ktrain.rows == Xtrain.rows && Ktrain.rows == Ktrain.cols)
          val lhs = Ktrain + (DenseMatrix.eye[Double](Ktrain.rows) :* lambda)
          println("starting A backslash b")
          val alphaStar = lhs \ Ytrain
          println("done with A backslash b")
          // evaluate training error-- we can do this now for DC-SVM since
          // the model used for prediction is the one associated with the center
          val predictions = MatrixUtils.matrixToRowArray(Ktrain * alphaStar).map(MaxClassifier.apply)
          val trainLabels = elems.map(_._1)
          assert(predictions.size == trainLabels.size)

          val nErrors = predictions.zip(trainLabels).filter { case (x, y) => x != y }.size

          val trainErrorRate = (nErrors.toDouble / trainLabels.size.toDouble)
          val trainAccRate = 1.0 - trainErrorRate

          println("[" + partId + "] localTrainEval acc: " + trainAccRate + ", err: " + trainErrorRate + ", nErrors: " + nErrors + ", nLocal: " + trainLabels.size)

          Iterator.single((alphaStar, Xtrain, trainLabels, predictions))
        }.cache()
    models.count()

  }


  def runSimple(conf: MnistDCSolverConfig): Unit = {

    logInfo(ccAsMap(conf).toString)

    val numClasses = 10
    val mnistImageSize = 784

    val train =
      MatrixUtils.matrixToRowArray(csvread(new File(conf.trainLocation), ','))
        .map(x => (x(0).toInt - 1, x(1 until x.length)))
        .map { case (y, x) => (y, (x * (2.0 / 255.0)) - 1.0) }

    // take first 10000

    val lambda = conf.lambda
    val gamma = conf.gamma
    val train10k = train.take(10000)

    val classLabeler = ClassLabelIndicatorsFromIntLabels(numClasses)
    val Xtrain = MatrixUtils.rowsToMatrix(train10k.map(_._2))
    val Ytrain = MatrixUtils.rowsToMatrix(train10k.map(_._1).map(classLabeler.apply))

    println("Xtrain.shape: " + Xtrain.rows + ", " + Xtrain.cols)
    println("Ytrain.shape: " + Ytrain.rows + ", " + Ytrain.cols)

    try {
      val Ktrain = GaussianKernel(gamma).apply(Xtrain)
      assert(Ktrain.rows == Xtrain.rows && Ktrain.rows == Ktrain.cols)
      val lhs = Ktrain + (DenseMatrix.eye[Double](Ktrain.rows) :* lambda)
      println("starting A backslash b")
      val alphaStar = lhs \ Ytrain
      println("done with A backslash b")
      // evaluate training error-- we can do this now for DC-SVM since
      // the model used for prediction is the one associated with the center

      val predictions = MatrixUtils.matrixToRowArray(Ktrain * alphaStar).map(MaxClassifier.apply)
      val trainLabels = train10k.map(_._1)
      assert(predictions.size == trainLabels.size)
      val nErrors = predictions.zip(trainLabels).filter { case (x, y) => x != y }.size

      //val predictions = Ktrain * alphaStar
      //println("done with predictions " + predictions.rows + " " + train10k.size)
      ////val predictions = MatrixUtils.matrixToRowArray(Ktrain * alphaStar).map(MaxClassifier.apply)
      //val trainLabels: Array[Int] = train10k.map(_._1).toArray
      //assert(predictions.rows == trainLabels.size)
      //println("predictions.rows " + predictions.rows + ", trainLabels.size" + trainLabels.size)
      //val nErrors = (0 until predictions.rows).map { i =>
      //  if (argmax(predictions(i, ::)) != trainLabels(i)) 1 else 0
      //}.sum

      val trainErrorRate = (nErrors.toDouble / train10k.size.toDouble)
      val trainAccRate = 1.0 - trainErrorRate

      println("localTrainEval acc: " + trainAccRate + ", err: " + trainErrorRate)
    } catch {
      case e: Exception =>
        println("WE GOT EXCEPTION", e)
        e.printStackTrace()
    }
  }


  def run(sc: SparkContext, conf: MnistDCSolverConfig): Pipeline[DenseVector[Double], Int] = {
    logInfo(ccAsMap(conf).toString)

    // This is a property of the MNIST Dataset (digits 0 - 9)
    val numClasses = 10

    val prng = new MersenneTwister(conf.seed)
    val kmeansSeed = prng.nextLong
    val randomSignSource = new RandBasis(new ThreadLocalRandomGenerator(prng))

    // The number of pixels in an MNIST image (28 x 28 = 784)
    // Because the mnistImageSize is 784, we get 512 PaddedFFT features per FFT.
    val mnistImageSize = 784

    val startTime = System.nanoTime()

    val train = LabeledData(
      CsvDataLoader(sc, conf.trainLocation, conf.numPartitions)
        // The pipeline expects 0-indexed class labels, but the labels in the file are 1-indexed
        .map(x => (x(0).toInt - 1, x(1 until x.length)))
        .map { case (y, x) => (y, (x * (2.0 / 255.0)) - 1.0) }
        .cache())

    val test = LabeledData(
      CsvDataLoader(sc, conf.testLocation, conf.numPartitions)
        // The pipeline expects 0-indexed class labels, but the labels in the file are 1-indexed
        .map(x => (x(0).toInt - 1, x(1 until x.length)))
        .map { case (y, x) => (y, (x * (2.0 / 255.0)) - 1.0) }
        .cache())

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

    val lambda = conf.lambda
    val gamma = conf.gamma

    val kmeansStartTime = System.nanoTime()
    val kmeans = conf.kmeansCenters.map { fname =>
      val centers = csvread(new File(fname), ',')
      KMeansModel(centers)
    }.getOrElse {
      val trainSubsample = train.data.sample(false, conf.kmeansSampleSize, kmeansSeed)
      KMeansPlusPlusEstimator(conf.numPartitions, 100).fit(trainSubsample)
    }
    logInfo(s"KMeans took ${(System.nanoTime() - kmeansStartTime)/1e9} s")

    val trainingStartTime = System.nanoTime()
    val trainAssignments: org.apache.spark.rdd.RDD[Int] = kmeans(train.data).map(oneHotToNumber)

    val trainPartitions: org.apache.spark.rdd.RDD[(Int, (Int, DenseVector[Double]))] = trainAssignments.zip(train.labeledData)
    val models: org.apache.spark.rdd.RDD[(Int, (DenseMatrix[Double], DenseMatrix[Double], Seq[Int], Seq[Int], Int))] = trainPartitions.groupByKey()
      .mapValues { partition =>
          val elems: Seq[(Int, DenseVector[Double])] = partition.toSeq
          println("elems.size: " + elems.size)
          val classLabeler = ClassLabelIndicatorsFromIntLabels(numClasses)
          val Xtrain = MatrixUtils.rowsToMatrix(elems.map(_._2))
          val Ytrain = MatrixUtils.rowsToMatrix(elems.map(_._1).map(classLabeler.apply))
          println("Xtrain.shape: " + Xtrain.rows + ", " + Xtrain.cols)
          println("Ytrain.shape: " + Ytrain.rows + ", " + Ytrain.cols)
          val Ktrain = GaussianKernel(gamma).apply(Xtrain)
          assert(Ktrain.rows == Xtrain.rows && Ktrain.rows == Ktrain.cols)
          val lhs = Ktrain + (DenseMatrix.eye[Double](Ktrain.rows) :* lambda)
          println("starting A backslash b")
          val alphaStar = lhs \ Ytrain
          println("done with A backslash b")
          // evaluate training error-- we can do this now for DC-SVM since
          // the model used for prediction is the one associated with the center
          val predictions = MatrixUtils.matrixToRowArray(Ktrain * alphaStar).map(MaxClassifier.apply).toSeq
          val trainLabels = elems.map(_._1)
          assert(predictions.size == trainLabels.size)

          val nErrors = predictions.zip(trainLabels).filter { case (x, y) => x != y }.size

          val trainErrorRate = (nErrors.toDouble / Xtrain.rows.toDouble)
          val trainAccRate = 1.0 - trainErrorRate

          println("localTrainEval acc: " + trainAccRate + ", err: " + trainErrorRate, ", nErrors: " + nErrors + ", nLocal: " + Xtrain.rows)

          (alphaStar, Xtrain, trainLabels, predictions, nErrors)
        }.cache()
    models.count()
    logInfo(s"Training took ${(System.nanoTime() - trainingStartTime)/1e9} s")

    val abc: Seq[Seq[Int]] = models.map(_._2._3).collect()
    val dfg: Seq[Seq[Int]] = models.map(_._2._4).collect()

    val flattenedTrainLabels = abc.flatten
    val flattenedTrainPredictions = dfg.flatten

    println("nTotalErrors: " + models.map(_._2._5).collect())

    assert(flattenedTrainLabels.size == flattenedTrainPredictions.size)
    assert(flattenedTrainLabels.size == train.labeledData.count)
    val nTrainErrors = flattenedTrainLabels.zip(flattenedTrainPredictions).filter { case (x, y) => x != y }.size
    val trainErrorRate = (nTrainErrors.toDouble / flattenedTrainLabels.size.toDouble)
    val trainAccRate = 1.0 - trainErrorRate
    println("trainEval acc: " + trainAccRate + ", err: " + trainErrorRate)

    val testData: org.apache.spark.rdd.RDD[(Int, Iterable[(Int, DenseVector[Double])])] = kmeans(test.data).map(oneHotToNumber).zip(test.labeledData).groupByKey(models.partitioner.get)

    val testEvaluationStartTime = System.nanoTime()
    val testEvaluation = models.join(testData).mapValues { case (lhs, rhs) =>
      val alphaStar = lhs._1
      val Xtrain = lhs._2
      val rhsSeq = rhs.toSeq
      val Xtest = MatrixUtils.rowsToMatrix(rhsSeq.map(_._2))
      val Ytest = rhsSeq.map(_._1)
      val Ktesttrain = GaussianKernel(gamma).apply(Xtest, Xtrain)
      val testPredictions = MatrixUtils.matrixToRowArray(Ktesttrain * alphaStar).map(MaxClassifier.apply)

      val nErrors = testPredictions.zip(Ytest).filter { case (x, y) => x != y }.size
      val testErrorRate = (nErrors.toDouble / Xtest.size.toDouble)
      val testAccRate = 1.0 - testErrorRate
      println("localtestEval acc: " + testAccRate + ", err: " + testErrorRate)


      (Ytest, testPredictions)
    }.cache()
    testEvaluation.count()
    logInfo(s"Test evaluation took ${(System.nanoTime() - testEvaluationStartTime)/1e9} s")

    // RDD does not have flatten?

    val flattenedTestLabels = testEvaluation.map(_._2._1).flatMap(x => x)
    val flattenedTestPredictions = testEvaluation.map(_._2._2).flatMap(x => x)

    //val trainEval = MulticlassClassifierEvaluator(flattenedTrainPredictions, flattenedTrainLabels, numClasses)
    //logInfo("TRAIN Error is " + (100 * trainEval.totalError) + "%")

    val testEval = MulticlassClassifierEvaluator(flattenedTestPredictions, flattenedTestLabels, numClasses)
    logInfo("TEST Error is " + (100 * testEval.totalError) + "%")

    val endTime = System.nanoTime()
    logInfo(s"Pipeline took ${(endTime - startTime)/1e9} s")
    null
  }

  case class MnistDCSolverConfig(
      trainLocation: String = "",
      testLocation: String = "",
      numPartitions: Int = 10,
      kmeansSampleSize: Double = 0.1,
      kmeansCenters: Option[String] = None,
      lambda: Double = 0.0,
      gamma: Double = 0,
      seed: Long = 0)

  def parse(args: Array[String]): MnistDCSolverConfig = new OptionParser[MnistDCSolverConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("numPartitions") required() action { (x,c) => c.copy(numPartitions=x) }
    opt[Double]("kmeansSampleSize") action { (x,c) => c.copy(kmeansSampleSize=x) }
    opt[String]("kmeansCenters") action { (x,c) => c.copy(kmeansCenters=Some(x)) }
    opt[Double]("lambda") required() action { (x,c) => c.copy(lambda=x) }
    opt[Double]("gamma") required() action { (x,c) => c.copy(gamma=x) }
    opt[Long]("seed") required() action { (x,c) => c.copy(seed=x) }
  }.parse(args, MnistDCSolverConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   *
   * @param args
   */
  def main(args: Array[String]) = {
    val appConfig = parse(args)

    //runSimple(appConfig)

    //val conf = new SparkConf().setAppName(appName)
    //conf.setIfMissing("spark.master", "local[24]")
    //val sc = new SparkContext(conf)
    //runSimpleSpark(sc, appConfig)
    //sc.stop()

    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[24]")
    val sc = new SparkContext(conf)
    run(sc, appConfig)
    sc.stop()
  }
}
