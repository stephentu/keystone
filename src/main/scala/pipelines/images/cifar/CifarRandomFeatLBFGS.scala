package pipelines.images.cifar

import breeze.linalg._
import breeze.numerics._

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import loaders.{CsvDataLoader, LabeledData}
import nodes.learning._
import nodes.stats.CosineRandomFeatures
import nodes.util.{Cacher, ClassLabelIndicatorsFromIntLabels}

import evaluation._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import pipelines._
import scopt.OptionParser
import utils.MatrixUtils
import workflow.Pipeline

import org.apache.spark.rdd.RDD

object CifarRandomFeatLBFGS extends Serializable with Logging {
  val appName = "CifarRandomFeatLBFGS"


  def testCb(testImageIds: RDD[String],
             testFeats: RDD[DenseVector[Double]],
             testActuals: RDD[Int],
             numClasses: Int,
             lm: LinearMapper[DenseVector[Double]]): Double = {
    val testPredictions = lm(testFeats)
    val testEval = AugmentedExamplesEvaluator(
        testImageIds, testPredictions, testActuals, numClasses)
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

  var WMat: DenseMatrix[Double] = null
  var BMat: DenseVector[Double] = null

  def getWandB(seed: Long, numInputFeatures: Int, numOutputFeatures: Int, gamma: Double) = synchronized {
    if (WMat == null || BMat == null) {
      val random = new java.util.Random(seed)
      val randomSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(random.nextLong())))
      val wDist = randomSource.gaussian
      val bDist = randomSource.uniform
      WMat = DenseMatrix.rand(numOutputFeatures, numInputFeatures, wDist)
      WMat :*= gamma
      BMat = DenseVector.rand(numOutputFeatures, bDist)
      BMat :*= ((2*math.Pi))
    }
    (WMat, BMat)
  }

  def randomCosineFeaturize(
      data: RDD[DenseVector[Double]],
      seed: Long,
      numInputFeatures: Int,
      numOutputFeatures: Int,
      gamma: Double): RDD[DenseMatrix[Double]] = {
    val featuresOut = data.mapPartitions { part =>
      if (part.hasNext) {
        val dataPart = MatrixUtils.rowsToMatrix(part)
        val (w, b) = getWandB(seed, numInputFeatures, numOutputFeatures, gamma)
        println(s"GOT dataPart dims ${dataPart.rows} x ${dataPart.cols}")
        println(s"GOT w dims ${w.rows} x ${w.cols}")
        val features = dataPart * w.t
        features(*, ::) :+= b
        cos.inPlace(features)
        Iterator.single(features)
      } else {
        Iterator.empty
      }
      //MatrixUtils.matrixToRowArray(features).iterator
    }
    featuresOut.cache()
    featuresOut
  }


  def run(sc: SparkContext, conf: CifarRandomFeatLBFGSConfig): Pipeline[DenseVector[Double], Int] = {
    println(ccAsMap(conf).toString)

    // This is a property of the CIFAR Dataset
    val numClasses = 10

    val startTime = System.nanoTime()
    val random = new java.util.Random(conf.seed)
    val randomSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(random.nextLong())))

    val train = LabeledData(
      materialize(
        sc.textFile(conf.trainLocation, conf.trainParts).map { row =>
          val parts = row.split(',')
          // ignore the image name
          val labelFeats = parts.tail.map(_.toDouble)
          (labelFeats(0).toInt, DenseVector(labelFeats.tail))
        }.repartition(conf.trainParts).cache(),
        "trainData"))

    val numInputFeats = train.data.first.size

    val testAll = materialize(sc.textFile(conf.testLocation, conf.testParts).map { row =>
      val parts = row.split(',')
      val labelFeats = parts.tail.map(_.toDouble)
      // include the image name
      (parts(0), labelFeats(0).toInt, DenseVector(labelFeats.tail))
    }.repartition(conf.testParts).cache(), "testData")

    val numBlocks = math.ceil(conf.numCosineFeatures.toDouble / conf.blockSize.toDouble).toInt

    // val featurizer = CosineRandomFeatures(
    //   numInputFeats,
    //   conf.numCosineFeatures,
    //   conf.cosineGamma,
    //   randomSource.gaussian,
    //   randomSource.uniform)

    val trainFeats = randomCosineFeaturize(train.data, conf.seed, numInputFeats, conf.numCosineFeatures, conf.cosineGamma)
    trainFeats.count

    val testFeats = randomCosineFeaturize(testAll.map(_._3), conf.seed, numInputFeats, conf.numCosineFeatures, conf.cosineGamma).mapPartitions { itr =>
      if (itr.hasNext) {
        MatrixUtils.matrixToRowArray(itr.next).iterator
      } else {
        Iterator.empty
      }
    }
    testFeats.count

    val trainLabVec = ClassLabelIndicatorsFromIntLabels(numClasses).apply(train.labels)
    val trainLabels = trainLabVec.mapPartitions { iter =>
      MatrixUtils.rowsToMatrixIter(iter)
    }

    val featTime = System.nanoTime()
    println(s"TIME_FEATURIZATION_${(featTime-startTime)/1e9}")

    val testCbBound = testCb(testAll.map(_._1), testFeats, testAll.map(_._2), numClasses, _: LinearMapper[DenseVector[Double]])
    if (conf.solver == "lbfgs") {
      val out = new BatchLBFGSwithL2(
        new LeastSquaresBatchGradient,
        numIterations=conf.numIters,
        regParam=conf.lambda,
        normStd=conf.normStd,
        epochCallback=Some(testCbBound),
        epochEveryTest=5).fitBatch(trainFeats, trainLabels)

      val model = LinearMapper[DenseVector[Double]](out._1, out._2, out._3)
      val testAcc = testCbBound(model)
      println(s"LAMBDA_${conf.lambda}_TEST_ACC_${testAcc}")
    } else if (conf.solver == "sgd") {
      val out = new MiniBatchSGDwithL2(
        new LeastSquaresBatchGradient,
        numIterations=conf.numIters,
        stepSize=conf.stepSize,
        regParam=conf.lambda,
        normStd=conf.normStd,
        epochCallback=Some(testCbBound),
        epochEveryTest=5).fitBatch(trainFeats, trainLabels)

      val model = LinearMapper[DenseVector[Double]](out._1, out._2, out._3)
      val testAcc = testCbBound(model)

    } else if (conf.solver == "bcd") {
      val trainFeatVec = trainFeats.flatMap(x => MatrixUtils.matrixToRowArray(x).iterator)
      val model = new BlockLeastSquaresEstimator(conf.blockSize, conf.numIters, conf.lambda, Some(conf.numCosineFeatures), normStd=conf.normStd, true).fit(trainFeatVec, trainLabVec)
      val testPredictions = (model).apply(testFeats)
      val testEval = AugmentedExamplesEvaluator(
         testAll.map(_._1), testPredictions, testAll.map(_._2), numClasses)
      val testAcc = (100 * testEval.totalAccuracy)
      //println("TEST MultiClass ACC " + testAcc)
      println(s"LAMBDA_${conf.lambda}_TEST_ACC_${testAcc}")
    } else {
      logError("solver not recognized")
    }
    val endTime = System.nanoTime()
    println(s"TIME_FULL_PIPELINE_${(endTime-startTime)/1e9}")

    null
  }

  case class CifarRandomFeatLBFGSConfig(
      trainLocation: String = "",
      testLocation: String = "",
      trainParts: Int = 0,
      testParts: Int = 0,
      lambda: Double = 0.0,
      numCosineFeatures: Int = 0,
      blockSize: Int = 0,
      cosineGamma: Double = 0,
      numIters: Int = 0,
      seed: Long = 0,
      normStd: Boolean = false,
      stepSize: Double = 0.0,
      solver: String = "")

  def parse(args: Array[String]): CifarRandomFeatLBFGSConfig = new OptionParser[CifarRandomFeatLBFGSConfig](appName) {

    private def isPositive(s: String)(x: Int) =
      if (x > 0) success else failure(s + " must be positive")

    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("trainParts") required() action { (x,c) => c.copy(trainParts=x) } validate isPositive("trainParts")
    opt[Int]("testParts") required() action { (x,c) => c.copy(testParts=x) } validate isPositive("testParts")
    opt[Double]("lambda") required() action { (x,c) => c.copy(lambda=x) }
    opt[Double]("cosineGamma") required() action { (x,c) => c.copy(cosineGamma=x) }
    opt[Long]("seed") required() action { (x,c) => c.copy(seed=x) }
    opt[Int]("numCosineFeatures") required() action { (x,c) => c.copy(numCosineFeatures=x) }
    opt[Int]("blockSize") required() action { (x,c) => c.copy(blockSize=x) }
    opt[Int]("numIters") required() action { (x,c) => c.copy(numIters=x) }
    opt[Boolean]("normStd") required() action { (x,c) => c.copy(normStd=x) }
    opt[Double]("stepSize") required() action { (x,c) => c.copy(stepSize=x) }
    opt[String]("solver") required() action { (x,c) => c.copy(solver=x) }
  }.parse(args, CifarRandomFeatLBFGSConfig()).get

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
    run(sc, appConfig)
    sc.stop()
  }
}

