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

    val featurizer = CosineRandomFeatures(
      numInputFeats,
      conf.numCosineFeatures,
      conf.cosineGamma,
      randomSource.gaussian,
      randomSource.uniform)

    val trainFeats = featurizer(train.data).cache()
    trainFeats.count

    val trainLabels = ClassLabelIndicatorsFromIntLabels(numClasses).apply(train.labels)

    val featTime = System.nanoTime()
    println(s"TIME_FEATURIZATION_${(featTime-startTime)/1e9}")

    if (conf.solver == "lbfgs") {
      val model = new BatchLBFGSwithL2(new LeastSquaresBatchGradient, numIterations=20, regParam=conf.lambda).fit(trainFeats, trainLabels)
      val testPredictions = model(testAll.map(_._3))
      val testEval = AugmentedExamplesEvaluator(
          testAll.map(_._1), testPredictions, testAll.map(_._2), numClasses)
      val testAcc = (100* testEval.totalAccuracy)
      println(s"LAMBDA_${conf.lambda}_TEST_ACC_${testAcc}")

      val endTime = System.nanoTime()
      println(s"TIME_FULL_PIPELINE_${(endTime-startTime)/1e9}")
    } else {
      logError("solver not recognized")
    }
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
    val sc = new SparkContext(conf)
    run(sc, appConfig)
    sc.stop()
  }
}

