package pipelines.images.cifar

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

import org.apache.spark.rdd.RDD

object CifarDCSolver extends Serializable with Logging {
  val appName = "CifarDCSolver"

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

  def run(sc: SparkContext, conf: CifarDCSolverConfig): Pipeline[DenseVector[Double], Int] = {
    logInfo(ccAsMap(conf).toString)

    // This is a property of the CIFAR Dataset
    val numClasses = 10

    val startTime = System.nanoTime()

    val train = LabeledData(
      materialize(
        sc.textFile(conf.trainLocation, conf.trainParts).map { row =>
          val parts = row.split(',')
          // ignore the image name
          val labelFeats = parts.tail.map(_.toDouble)
          (labelFeats(0).toInt, DenseVector(labelFeats.tail))
        }.repartition(conf.trainParts).cache(),
        "trainData"))

    val testAll = materialize(sc.textFile(conf.testLocation, conf.testParts).map { row =>
      val parts = row.split(',')
      val labelFeats = parts.tail.map(_.toDouble)
      // include the image name
      (parts(0), labelFeats(0).toInt, DenseVector(labelFeats.tail))
    }.repartition(conf.testParts), "testData")
    
    if (conf.useYuchen) {
      val dcsolver = DCSolverYuchen.fit(
        train, numClasses, conf.lambdas, conf.gamma, conf.numPartitions, conf.seed)
    
      conf.lambdas.zip(dcsolver.trainEvals.zip(dcsolver.augmentedMetrics(testAll,numClasses))).foreach { case (lambda, (trainEval, testEval)) =>
        trainAcc = (100 * trainEval.totalAccuracy)
        testAcc = (100* testEval.totalAccuracy)
        println(s"LAMBDA_${lambda}_TRAIN_ACC_${trainAcc}_TEST_ACC_${testAcc}")
      }
      val endTime = System.nanoTime()
      println(s"TIME_FULL_PIPELINE_${(endTime-startTime)/1e9}")
    } else {
      val dcsolver = DCSolver.fit(
        train, numClasses, conf.lambdas, conf.gamma, conf.numModels, conf.kmeansSampleSize, conf.seed)
    
      conf.lambdas.zip(dcsolver.trainEvals.zip(dcsolver.augmentedMetrics(testAll,numClasses))).foreach { case (lambda, (trainEval, testEval)) =>
        trainAcc = (100 * trainEval.totalAccuracy)
        testAcc = (100* testEval.totalAccuracy)
        println(s"LAMBDA_${lambda}_TRAIN_ACC_${trainAcc}_TEST_ACC_${testAcc}")
      }
      val endTime = System.nanoTime()
      println(s"TIME_FULL_PIPELINE_${(endTime-startTime)/1e9}")
    }
    null
  }

  case class CifarDCSolverConfig(
      trainLocation: String = "",
      testLocation: String = "",
      trainParts: Int = 0,
      testParts: Int = 0,
      numModels: Int = 10,
      numPartitions: Int = 128,
      kmeansSampleSize: Double = 0.1,
      lambdas: Seq[Double] = Seq.empty,
      gamma: Double = 0,
      seed: Long = 0)

  def parse(args: Array[String]): CifarDCSolverConfig = new OptionParser[CifarDCSolverConfig](appName) {

    private def isPositive(s: String)(x: Int) =
      if (x > 0) success else failure(s + " must be positive")

    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("trainParts") required() action { (x,c) => c.copy(trainParts=x) } validate isPositive("trainParts")
    opt[Int]("testParts") required() action { (x,c) => c.copy(testParts=x) } validate isPositive("testParts")
    opt[Int]("numModels") action { (x,c) => c.copy(numModels=x) } validate isPositive("numModels")
    opt[Int]("numPartitions") action { (x,c) => c.copy(numPartitions=x) } validate isPositive("numPartitions")
    opt[Double]("kmeansSampleSize") action { (x,c) => c.copy(kmeansSampleSize=x) }
    opt[Seq[Double]]("lambdas") required() action { (x,c) => c.copy(lambdas=x) }
    opt[Double]("gamma") required() action { (x,c) => c.copy(gamma=x) }
    opt[Long]("seed") required() action { (x,c) => c.copy(seed=x) }
  }.parse(args, CifarDCSolverConfig()).get

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

