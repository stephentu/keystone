package pipelines.speech.timit

import breeze.linalg._
import breeze.numerics._

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import nodes.learning._
import nodes.stats.CosineRandomFeatures
import nodes.util.{Cacher, ClassLabelIndicatorsFromIntLabels, VectorCombiner}
import nodes.util.MaxClassifier
import loaders.{CsvDataLoader, LabeledData, TimitFeaturesDataLoader}

import evaluation._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import pipelines._
import scopt.OptionParser
import utils.MatrixUtils
import workflow.Pipeline

import org.apache.spark.rdd.RDD

object TimitRandomFeatLBFGS extends Serializable with Logging {
  val appName = "TimitRandomFeatLBFGS"

  def testCb(testFeats: RDD[DenseMatrix[Double]],
             testActuals: RDD[Int],
             numClasses: Int,
             lm: LinearMapper[DenseVector[Double]]): Double = {

    val testPreds = lm.applyBatch(testFeats)
    val testPredictions = MaxClassifier.apply(testPreds)
    val testEval = MulticlassClassifierEvaluator(
        testPredictions, testActuals, numClasses)
    val testAcc = (100* testEval.totalAccuracy)
    testAcc
  }

  // http://stackoverflow.com/questions/1226555/case-class-to-map-in-scala
  private def ccAsMap(cc: AnyRef) =
    (Map[String, Any]() /: cc.getClass.getDeclaredFields) {(a, f) =>
      f.setAccessible(true)
      a + (f.getName -> f.get(cc))
    }

  private def materialize[T](rdd: RDD[T], s: String): RDD[T] = {
    rdd.setName(s)
    rdd.count()
    rdd
  }

  def randomCosineFeaturize(
      data: RDD[DenseVector[Double]],
      seed: Long,
      numInputFeatures: Int,
      numOutputFeatures: Int,
      gamma: Double): RDD[DenseMatrix[Double]] = {
    val featuresOut = data.mapPartitions { part =>
      if (part.hasNext) {
        val random = new java.util.Random(seed)
        val randomSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(random.nextLong())))
        val wDist = randomSource.gaussian
        val bDist = randomSource.uniform
        val W = DenseMatrix.rand(numOutputFeatures, numInputFeatures, wDist) :* gamma
        val b = DenseVector.rand(numOutputFeatures, bDist) :* ((2*math.Pi))

        val dataPart = MatrixUtils.rowsToMatrix(part)
        val features = dataPart * W.t
        features(*, ::) :+= b
        cos.inPlace(features)
        Iterator.single(features)
      } else {
        Iterator.empty
      }
      // MatrixUtils.matrixToRowArray(features).iterator
    }
    featuresOut.setName("featuresOut")
    featuresOut.cache()
    featuresOut
  }

  def run(sc: SparkContext, conf: TimitRandomFeatLBFGSConfig): Pipeline[DenseVector[Double], Int] = {
    println(ccAsMap(conf).toString)

    // This is a property of the Timit Dataset
    val numClasses = 147

    val startTime = System.nanoTime()
    val random = new java.util.Random(conf.seed)
    val randomSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(random.nextLong())))

    // Load the data
    val timitFeaturesData = TimitFeaturesDataLoader(
      sc,
      conf.trainDataLocation,
      conf.trainLabelsLocation,
      conf.testDataLocation,
      conf.testLabelsLocation,
      conf.trainParts)

    val trainData = timitFeaturesData.train.data.cache().setName("trainRaw")
    trainData.count()
    val trainLabels = timitFeaturesData.train.labels.cache().setName("trainLabels")
    val train = LabeledData(trainLabels.zip(trainData).repartition(conf.trainParts).cache().setName("trainRepart"))

    train.data.count
    train.labels.count
   
    val testData = timitFeaturesData.test.data.cache().setName("testRaw")
    val numTest = testData.count()
    val testLabels = timitFeaturesData.test.labels.cache().setName("testLabels")
    val test = LabeledData(testLabels.zip(testData))

    test.data.count
    test.labels.count

    val numInputFeats = train.data.first.size
    val numBlocks = math.ceil(conf.numCosineFeatures.toDouble / conf.blockSize.toDouble).toInt

		val trainFeats = randomCosineFeaturize(train.data, conf.seed, numInputFeats, conf.numCosineFeatures, conf.cosineGamma)
    // val trainFeats = featurizer(train.data).cache()
    trainFeats.count

    //val testFeats = featurizer(test.data).cache()
		val testFeats = randomCosineFeaturize(test.data, conf.seed, numInputFeats, conf.numCosineFeatures, conf.cosineGamma)
    testFeats.count

    val trainLabelsVec = ClassLabelIndicatorsFromIntLabels(numClasses).apply(train.labels)
    val trainLabelsMat = trainLabelsVec.mapPartitions { iter =>
      MatrixUtils.rowsToMatrixIter(iter)
    }

    val featTime = System.nanoTime()
    println(s"TIME_FEATURIZATION_${(featTime-startTime)/1e9}")

    val testCbBound = testCb(testFeats, testLabels, numClasses, _: LinearMapper[DenseVector[Double]])
    if (conf.solver == "lbfgs") {
      val out = new BatchLBFGSwithL2(
        new LeastSquaresBatchGradient,
        numIterations=conf.numIters,
        regParam=conf.lambda,
        epochCallback=Some(testCbBound),
        epochEveryTest=5).fitBatch(trainFeats, trainLabelsMat)
      val model = LinearMapper[DenseVector[Double]](out._1, out._2, out._3)
      val testAcc = testCbBound(model)
      println(s"LAMBDA_${conf.lambda}_TEST_ACC_${testAcc}")
    } else if (conf.solver == "sgd") {
      val out = new MiniBatchSGDwithL2(
        new LeastSquaresBatchGradient,
        numIterations=conf.numIters,
        stepSize=conf.stepSize,
        dampen=conf.sgdDampen,
        regParam=conf.lambda,
        miniBatchFraction=conf.miniBatchFraction,
        epochCallback=Some(testCbBound),
        epochEveryTest=1).fitBatch(trainFeats, trainLabelsMat)
      val model = LinearMapper[DenseVector[Double]](out._1, out._2, out._3)
      val testAcc = testCbBound(model)
      println(s"LAMBDA_${conf.lambda}_TEST_ACC_${testAcc}")
    } else if (conf.solver == "cocoa") {
      val out = new CocoaSDCAwithL2(
        new LeastSquaresBatchGradient,
        numIterations=conf.numIters,
        regParam=conf.lambda,
        numLocalItersFraction=conf.cocoaLocalItersFraction,
        beta=conf.cocoaBeta,
        normRows=false,
        epochCallback=Some(testCbBound),
        epochEveryTest=1).fitBatch(trainFeats, trainLabelsMat)

      val model = LinearMapper[DenseVector[Double]](out._1, out._2, out._3)
      val testAcc = testCbBound(model)
      println(s"LAMBDA_${conf.lambda}_TEST_ACC_${testAcc}")
    } else if (conf.solver == "bcd") {
      val trainFeatVec = trainFeats.flatMap(x => MatrixUtils.matrixToRowArray(x).iterator)
      val model = new BlockLeastSquaresEstimator(
        conf.blockSize, conf.numIters, conf.lambda, Some(conf.numCosineFeatures), true).fit(
          trainFeatVec, trainLabelsVec)
        val testPredictions = (model andThen MaxClassifier).apply(testFeats.mapPartitions(itr => MatrixUtils.matrixToRowArray(itr.next).iterator))
      val testEval = MulticlassClassifierEvaluator(
          testPredictions, testLabels, numClasses)
      val testAcc = (100* testEval.totalAccuracy)
      println(s"LAMBDA_${conf.lambda}_TEST_ACC_${testAcc}")
    } else {
      logError("solver not recognized")
    }
    val endTime = System.nanoTime()
    println(s"TIME_FULL_PIPELINE_${(endTime-startTime)/1e9}")
    null
  }

  case class TimitRandomFeatLBFGSConfig(
      trainDataLocation: String = "",
      trainLabelsLocation: String = "",
      testDataLocation: String = "",
      testLabelsLocation: String = "",
      trainParts: Int = 0,
      lambda: Double = 0.0,
      numCosineFeatures: Int = 0,
      blockSize: Int = 0,
      cosineGamma: Double = 0,
      numIters: Int = 0,
      seed: Long = 0,
      sgdDampen: Option[Double] = None,
      stepSize: Double = 0.0,
      miniBatchFraction: Double = 0.0,
      cocoaBeta: Double = 0.0,
      cocoaLocalItersFraction: Double = 0.0,
      solver: String = "")

  def parse(args: Array[String]): TimitRandomFeatLBFGSConfig = new OptionParser[TimitRandomFeatLBFGSConfig](appName) {

    private def isPositive(s: String)(x: Int) =
      if (x > 0) success else failure(s + " must be positive")

    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainDataLocation") required() action { (x,c) => c.copy(trainDataLocation=x) }
    opt[String]("trainLabelsLocation") required() action { (x,c) => c.copy(trainLabelsLocation=x) }
    opt[String]("testDataLocation") required() action { (x,c) => c.copy(testDataLocation=x) }
    opt[String]("testLabelsLocation") required() action { (x,c) => c.copy(testLabelsLocation=x) }
    opt[Int]("trainParts") required() action { (x,c) => c.copy(trainParts=x) } validate isPositive("trainParts")
    opt[Double]("lambda") required() action { (x,c) => c.copy(lambda=x) }
    opt[Double]("cosineGamma") required() action { (x,c) => c.copy(cosineGamma=x) }
    opt[Long]("seed") required() action { (x,c) => c.copy(seed=x) }
    opt[Int]("numCosineFeatures") required() action { (x,c) => c.copy(numCosineFeatures=x) }
    opt[Int]("blockSize") required() action { (x,c) => c.copy(blockSize=x) }
    opt[Int]("numIters") required() action { (x,c) => c.copy(numIters=x) }
    opt[Double]("stepSize") required() action { (x,c) => c.copy(stepSize=x) }
    opt[Double]("sgdDampen") action { (x,c) => c.copy(sgdDampen=Some(x)) }
    opt[Double]("miniBatchFraction") required() action { (x,c) => c.copy(miniBatchFraction=x) }
    opt[Double]("cocoaBeta") required() action { (x,c) => c.copy(cocoaBeta=x) }
    opt[Double]("cocoaLocalItersFraction") required() action { (x,c) => c.copy(cocoaLocalItersFraction=x) }
    opt[String]("solver") required() action { (x,c) => c.copy(solver=x) }
  }.parse(args, TimitRandomFeatLBFGSConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   *
   * @param args
   */
  def main(args: Array[String]) = {
    val appConfig = parse(args)
    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[24]")
    conf.remove("spark.jars")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("/tmp/spark-checkpoint")
    run(sc, appConfig)
    sc.stop()
  }
}

