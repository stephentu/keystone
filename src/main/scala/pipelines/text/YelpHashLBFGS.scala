package pipelines.text

import breeze.linalg._
import breeze.numerics._

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}

import scala.collection.mutable.{HashMap, HashSet}
import scala.collection.mutable.ArrayBuffer

import loaders.{CsvDataLoader, LabeledData}
import nodes.learning._
import evaluation._
import nodes.util.{ClassLabelIndicatorsFromIntLabels, MaxClassifier}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import pipelines._
import scopt.OptionParser
import utils.MatrixUtils
import workflow.Pipeline

import org.apache.spark.rdd.RDD

object YelpHashLBFGS extends Serializable with Logging {
  val appName = "YelpHashLBFGS"

  // http://stackoverflow.com/questions/1226555/case-class-to-map-in-scala
  private def ccAsMap(cc: AnyRef) =
    (Map[String, Any]() /: cc.getClass.getDeclaredFields) {(a, f) =>
      f.setAccessible(true)
      a + (f.getName -> f.get(cc))
    }

  def testCb(testFeats: RDD[DenseVector[Double]],
             testActuals: RDD[Int],
             numClasses: Int,
             lm: LinearMapper[DenseVector[Double]]): Double = {
    val testPredictions = (lm andThen MaxClassifier).apply(testFeats)
    val testEval = MulticlassClassifierEvaluator(
       testPredictions, testActuals, numClasses)
    val testAcc = (100* testEval.totalAccuracy)
    testAcc
  }

  private def materialize[T](rdd: RDD[T], s: String): RDD[T] = {
    rdd.setName(s)
    rdd.count()
    rdd
  }

  private def nonNegativeMod(x: Int, mod: Int): Int = {
    val rawMod = x % mod
    rawMod + (if (rawMod < 0) mod else 0)
  }

  private def hashTokens(tokens: Seq[String], numOutputFeatures: Int): DenseVector[Double] = {
    val ret = DenseVector.zeros[Double](numOutputFeatures)

    tokens.foreach { token =>
      val h = util.hashing.MurmurHash3.stringHash(token)
      val idx = nonNegativeMod(h, numOutputFeatures)
      val shiftedIdx = idx
      ret(shiftedIdx) += (if (h >= 0) 1.0 else -1.0)
    }

    // TODO(shivaram): This is not consistent with the string
    // kernel generator. But seems to work better in practice ??
    //
    ret :/= tokens.size.toDouble
    ret
  }

  private def tokensToNGrams(tokens: Seq[String], nGrams: Int): Seq[String] = {
    val origTokens = tokens
    val newTokens = ArrayBuffer.empty[String]
    (1 to min(nGrams, origTokens.size)).foreach { n =>
      (0 to (origTokens.size - n)).foreach { i =>
        newTokens += origTokens.slice(i, i + n).mkString("_")
      }
    }
    origTokens ++ newTokens
  }

  private def hashFeaturizeReviews(
      trainFeatures: RDD[Array[String]],
      testFeatures: RDD[Array[String]],
      numNgrams: Int,
      numOutputFeatures: Int) = {

    val startTime = System.nanoTime

    // step 1: hash the training 3-grams on the fly for training
    val trainHashStartTime = System.nanoTime
    val trainFeaturesHashed = trainFeatures
      .map(tokens => tokensToNGrams(tokens.toSeq, numNgrams))
      .map(tokens => hashTokens(tokens, numOutputFeatures))
      .cache().setName("trainFeaturesHashed")
    val nTrain = trainFeaturesHashed.count
    println(s"training ngrams took ${(System.nanoTime-trainHashStartTime)/1e9}")

    // step 2: hash the testing 3-grams on the fly for testing
    val testHashStartTime = System.nanoTime
    val testFeaturesHashed = testFeatures
      .map(tokens => tokensToNGrams(tokens.toSeq, numNgrams))
      .map(tokens => hashTokens(tokens, numOutputFeatures))
      .cache().setName("testFeaturesHashed")
    testFeaturesHashed.count
    println(s"testing ngrams took ${(System.nanoTime-testHashStartTime)/1e9}")

    (trainFeaturesHashed, testFeaturesHashed)
  }

  def run(sc: SparkContext, conf: YelpHashLBFGSConfig): Pipeline[DenseVector[Double], Int] = {
    println(ccAsMap(conf).toString)

    // This is a property of the Yelp Dataset
    val numClasses = 5
    val numNgrams = 3
    val startTime = System.nanoTime()

    val trainStr = sc.textFile(conf.trainDataLocation, conf.trainParts).map { row =>
      row.split(' ')
    }
    // labels encoded in {1, 2, 3, 4, 5}
    val trainLab = sc.textFile(conf.trainLabelsLocation, conf.trainParts).map { row =>
      row.toInt - 1
    }
    val trainStrLab = materialize(
        trainStr.zip(trainLab).repartition(conf.trainParts), "trainData")

    val testStr = sc.textFile(conf.testDataLocation, conf.testParts).map { row =>
      row.split(' ')
    }
    // labels encoded in {1, 2, 3, 4, 5}
    val testLab = sc.textFile(conf.testLabelsLocation, conf.testParts).map { row =>
      row.toInt - 1
    }
    val testStrLab = materialize(
        testStr.zip(testLab).repartition(conf.testParts), "testData")

    val (trainFeat, testFeat) = hashFeaturizeReviews(trainStrLab.map(_._1), testStrLab.map(_._1), numNgrams, conf.numHashFeatures)

    // TODO: Do this before cache ??
    val trainFeatMat = trainFeat.mapPartitions { iter =>
      MatrixUtils.rowsToMatrixIter(iter)
    }  

    val trainLabMat = ClassLabelIndicatorsFromIntLabels(numClasses).apply(trainStrLab.map(_._2)).mapPartitions { iter =>
      MatrixUtils.rowsToMatrixIter(iter)
    }

    if (conf.solver == "lbfgs") {
      val testCbBound = testCb(testFeat, testStrLab.map(_._2), numClasses, _: LinearMapper[DenseVector[Double]])

      val out = new BatchLBFGSwithL2(new LeastSquaresBatchGradient, numIterations=conf.numIters, regParam=conf.lambda, epochCallback=Some(testCbBound), epochEveryTest=10).fitBatch(trainFeatMat, trainLabMat)

      val model = LinearMapper[DenseVector[Double]](out._1, out._2, out._3)
      val testAcc = testCbBound(model)
      println(s"LAMBDA_${conf.lambda}_TEST_ACC_${testAcc}")

      val endTime = System.nanoTime()
      println(s"TIME_FULL_PIPELINE_${(endTime-startTime)/1e9}")
    } else {
      logError("solver not recognized")
    }
    null
  }

  case class YelpHashLBFGSConfig(
      trainDataLocation: String = "",
      trainLabelsLocation: String = "",
      testDataLocation: String = "",
      testLabelsLocation: String = "",
      trainParts: Int = 0,
      testParts: Int = 0,
      numHashFeatures: Int = 0,
      numIters: Int = 0,
      lambda: Double = 0.0,
      seed: Long = 0,
      solver: String = "")

  def parse(args: Array[String]): YelpHashLBFGSConfig = new OptionParser[YelpHashLBFGSConfig](appName) {

    private def isPositive(s: String)(x: Int) =
      if (x > 0) success else failure(s + " must be positive")

    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainDataLocation") required() action { (x,c) => c.copy(trainDataLocation=x) }
    opt[String]("trainLabelsLocation") required() action { (x,c) => c.copy(trainLabelsLocation=x) }
    opt[String]("testDataLocation") required() action { (x,c) => c.copy(testDataLocation=x) }
    opt[String]("testLabelsLocation") required() action { (x,c) => c.copy(testLabelsLocation=x) }
    opt[Int]("trainParts") required() action { (x,c) => c.copy(trainParts=x) } validate isPositive("trainParts")
    opt[Int]("testParts") required() action { (x,c) => c.copy(testParts=x) } validate isPositive("testParts")
    opt[Int]("numHashFeatures") required() action { (x,c) => c.copy(numHashFeatures=x) }
    opt[Int]("numIters") required() action { (x,c) => c.copy(numIters=x) }
    opt[Double]("lambda") required() action { (x,c) => c.copy(lambda=x) }
    opt[Long]("seed") required() action { (x,c) => c.copy(seed=x) }
    opt[String]("solver") required() action { (x,c) => c.copy(solver=x) }
  }.parse(args, YelpHashLBFGSConfig()).get

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

