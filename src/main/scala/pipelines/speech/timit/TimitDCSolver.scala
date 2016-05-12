package pipelines.speech.timit

import breeze.linalg._
import breeze.numerics._

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import loaders.{CsvDataLoader, LabeledData, TimitFeaturesDataLoader}
import nodes.learning.{DCSolver, DCSolverYuchen}
import nodes.util.ClassLabelIndicatorsFromIntLabels
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import pipelines._
import scopt.OptionParser
import utils.MatrixUtils
import workflow.Pipeline

import org.apache.spark.rdd.RDD

object TimitDCSolver extends Serializable with Logging {
  val appName = "TimitDCSolver"

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

  def run(sc: SparkContext, conf: TimitDCSolverConfig): Pipeline[DenseVector[Double], Int] = {
    println(ccAsMap(conf).toString)

    // This is a property of the Timit Dataset
    val numClasses = 147

    val startTime = System.nanoTime()

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
    val train = LabeledData(trainLabels.zip(trainData))
   
    val testData = timitFeaturesData.test.data.cache().setName("testRaw")
    val numTest = testData.count()
    val testLabels = timitFeaturesData.test.labels.cache().setName("testLabels")
    val test = LabeledData(testLabels.zip(testData))
    
    if (conf.solver == "dcyuchen") {
      val dcsolver = DCSolverYuchen.fit(
        train, numClasses, conf.lambdas, conf.gamma, conf.numPartitions, conf.seed)
    
      conf.lambdas.zip(dcsolver.metrics(test, numClasses)).foreach { case (lambda, testEval) =>
        val testAcc = (100* testEval.totalAccuracy)
        println(s"LAMBDA_${lambda}_TEST_ACC_${testAcc}")
      }
      val endTime = System.nanoTime()
      println(s"TIME_FULL_PIPELINE_${(endTime-startTime)/1e9}")
    } else if (conf.solver == "dcsvm"){
      val dcsolver = DCSolver.fit(
        train, numClasses, conf.lambdas, conf.gamma, conf.numModels, conf.kmeansSampleSize, conf.seed)
    
      conf.lambdas.zip(dcsolver.trainEvals.zip(dcsolver.metrics(test,numClasses))).foreach { case (lambda, (trainEval, testEval)) =>
        val trainAcc = (100 * trainEval.totalAccuracy)
        val testAcc = (100* testEval.totalAccuracy)
        println(s"LAMBDA_${lambda}_TRAIN_ACC_${trainAcc}_TEST_ACC_${testAcc}")
      }
      val endTime = System.nanoTime()
      println(s"TIME_FULL_PIPELINE_${(endTime-startTime)/1e9}")
    } else {
      logError("solver not recognized")
    }
    null
  }

  case class TimitDCSolverConfig(
      trainDataLocation: String = "",
      trainLabelsLocation: String = "",
      testDataLocation: String = "",
      testLabelsLocation: String = "",
      trainParts: Int = 0,
      numModels: Int = 10,
      numPartitions: Int = 128,
      kmeansSampleSize: Double = 0.1,
      lambdas: Seq[Double] = Seq.empty,
      gamma: Double = 0,
      seed: Long = 0,
      solver: String = "")

  def parse(args: Array[String]): TimitDCSolverConfig = new OptionParser[TimitDCSolverConfig](appName) {

    private def isPositive(s: String)(x: Int) =
      if (x > 0) success else failure(s + " must be positive")

    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainDataLocation") required() action { (x,c) => c.copy(trainDataLocation=x) }
    opt[String]("trainLabelsLocation") required() action { (x,c) => c.copy(trainLabelsLocation=x) }
    opt[String]("testDataLocation") required() action { (x,c) => c.copy(testDataLocation=x) }
    opt[String]("testLabelsLocation") required() action { (x,c) => c.copy(testLabelsLocation=x) }
    opt[Int]("trainParts") required() action { (x,c) => c.copy(trainParts=x) } validate isPositive("trainParts")
    opt[Int]("numModels") action { (x,c) => c.copy(numModels=x) } validate isPositive("numModels")
    opt[Int]("numPartitions") action { (x,c) => c.copy(numPartitions=x) } validate isPositive("numPartitions")
    opt[Double]("kmeansSampleSize") action { (x,c) => c.copy(kmeansSampleSize=x) }
    opt[Seq[Double]]("lambdas") required() action { (x,c) => c.copy(lambdas=x) }
    opt[Double]("gamma") required() action { (x,c) => c.copy(gamma=x) }
    opt[Long]("seed") required() action { (x,c) => c.copy(seed=x) }
    opt[String]("solver") required() action { (x,c) => c.copy(solver=x) }
  }.parse(args, TimitDCSolverConfig()).get

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

