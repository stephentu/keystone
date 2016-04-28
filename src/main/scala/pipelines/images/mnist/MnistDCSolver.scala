package pipelines.images.mnist

import breeze.linalg._
import breeze.numerics._

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import loaders.{CsvDataLoader, LabeledData}
import nodes.learning.DCSolver
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import pipelines._
import scopt.OptionParser
import utils.MatrixUtils
import workflow.Pipeline


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

    val dcsolver = DCSolver.fit(
      train, numClasses, conf.lambda, conf.gamma, conf.numPartitions, conf.kmeansSampleSize, kmeansSeed)

    val trainEval = dcsolver.trainEval
    logInfo("TRAIN Acc is " + (100 * trainEval.totalAccuracy) + "%")
    logInfo("TRAIN Error is " + (100 * trainEval.totalError) + "%")

    val testEval = dcsolver.metrics(test, numClasses)
    logInfo("TEST Acc is " + (100 * testEval.totalAccuracy) + "%")
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
    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[24]")
    val sc = new SparkContext(conf)
    run(sc, appConfig)
    sc.stop()
  }
}
