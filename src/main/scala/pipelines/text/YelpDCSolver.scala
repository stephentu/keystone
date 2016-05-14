package pipelines.text

import breeze.linalg._
import breeze.numerics._

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}

import scala.collection.mutable.{HashMap, HashSet}
import scala.collection.mutable.ArrayBuffer

import loaders.{CsvDataLoader, LabeledData}
import nodes.learning.DCSolverYuchenSparse
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import pipelines._
import scopt.OptionParser
import utils.MatrixUtils
import workflow.Pipeline

import org.apache.spark.rdd.RDD

object YelpDCSolver extends Serializable with Logging {
  val appName = "YelpDCSolver"

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

  private def featurizeReviews(
      trainFeatures: RDD[Array[String]],
      testFeatures: RDD[Array[String]],
      numNgrams: Int) = {
    val startTime = System.nanoTime

    // step 1: build n-grams
    val trainFeaturesNGrams = trainFeatures.map { tokens =>
      tokensToNGrams(tokens.toSeq, numNgrams)
    }

    // step 2: convert tokens to unique IDs
    val globalDict = new java.util.HashMap[String, Long]
    trainFeaturesNGrams.mapPartitions { part =>
      val hs = new HashSet[String]
      part.foreach { tokens => hs ++= tokens }
      Iterator.single(hs)
    }.collect().foreach { hs =>
      hs.foreach { term => 
        if (!globalDict.containsKey(term)) {
          globalDict.put(term, globalDict.size)
        }
      }
    }

    println(s"Took ${(System.nanoTime - startTime)/1e9} seconds to build globalDict")
    println(s"\tglobalDict has ${globalDict.size} unique tokens")

    // step 3: featurize train/test data using these unique IDs

    val featurizeTrainStartTime = System.nanoTime
    println("DEBUG_Global dict size is " + globalDict.size)
    val globalDictBC = trainFeaturesNGrams.context.broadcast(globalDict)
    val trainFeaturesFinal = trainFeaturesNGrams.map { tokens =>
      val freqs = new HashMap[String, Int]
      val globalDict = globalDictBC.value
      tokens.foreach { token =>
        freqs.put(token, freqs.getOrElse(token, 0) + 1)
      }
      // count.toDouble / tokens.size.toDouble
      // TODO(shivaram): Figure out why the normalization breaks the Nystrom solve
      val features = freqs
        .map { case (token, count) => (globalDict.get(token), count.toDouble) }
        .toSeq.sortBy(_._1)

      // use two arrays so we can specialize on int, double
      (features.map(_._1).toArray, features.map(_._2).toArray)
    }.setName("trainFeaturesFinal").cache()
    trainFeaturesFinal.count

    println(s"Featurizing train took ${(System.nanoTime-featurizeTrainStartTime)/1e9} seoncds")

    val featurizeTestStartTime = System.nanoTime

    val testFeaturesFinal = testFeatures.map { tokens =>
      val freqs = new HashMap[String, Int]
      val globalDict = globalDictBC.value
      tokensToNGrams(tokens, numNgrams).foreach { token =>
        freqs.put(token, freqs.getOrElse(token, 0) + 1)
      }
      // tokens.size.toDouble
      // TODO(shivaram): Figure out why the normalization breaks the Nystrom solve
      val features = freqs
        .filter { case (token, _) => globalDict.containsKey(token) }
        .map { case (token, count) => (globalDict.get(token), count.toDouble) }
        .toSeq.sortBy(_._1)

      // use two arrays so we can specialize on int, double
      (features.map(_._1).toArray, features.map(_._2).toArray)
    }.setName("testFeaturesFinal").cache()
    testFeaturesFinal.count

    println(s"Featurizing test took ${(System.nanoTime-featurizeTestStartTime)/1e9} seoncds")

    globalDictBC.unpersist()

    (trainFeaturesFinal, testFeaturesFinal)
  }

  def run(sc: SparkContext, conf: YelpDCSolverConfig): Pipeline[DenseVector[Double], Int] = {
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

    val (trainFeat, testFeat) = featurizeReviews(trainStrLab.map(_._1), testStrLab.map(_._1), numNgrams) 
  
    val trainLabFeat = trainStrLab.map(_._2).zip(trainFeat)
    val testLabFeat = testStrLab.map(_._2).zip(testFeat)

    if (conf.solver == "dcyuchen") {
      val dcsolver = DCSolverYuchenSparse.fit(
        trainLabFeat, numClasses, conf.lambdas, conf.gamma, conf.numPartitions, conf.seed)
    
      conf.lambdas.zip(dcsolver.metrics(testLabFeat, numClasses)).foreach { case (lambda, testEval) =>
        val testAcc = (100* testEval.totalAccuracy)
        println(s"LAMBDA_${lambda}_TEST_ACC_${testAcc}")
      }
      val endTime = System.nanoTime()
      println(s"TIME_FULL_PIPELINE_${(endTime-startTime)/1e9}")
    } else {
      logError("solver not recognized")
    }
    null
  }

  case class YelpDCSolverConfig(
      trainDataLocation: String = "",
      trainLabelsLocation: String = "",
      testDataLocation: String = "",
      testLabelsLocation: String = "",
      trainParts: Int = 0,
      testParts: Int = 0,
      numModels: Int = 10,
      numPartitions: Int = 128,
      kmeansSampleSize: Double = 0.1,
      lambdas: Seq[Double] = Seq.empty,
      gamma: Double = 0,
      seed: Long = 0,
      solver: String = "")

  def parse(args: Array[String]): YelpDCSolverConfig = new OptionParser[YelpDCSolverConfig](appName) {

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
    opt[Int]("numModels") action { (x,c) => c.copy(numModels=x) } validate isPositive("numModels")
    opt[Int]("numPartitions") action { (x,c) => c.copy(numPartitions=x) } validate isPositive("numPartitions")
    opt[Double]("kmeansSampleSize") action { (x,c) => c.copy(kmeansSampleSize=x) }
    opt[Seq[Double]]("lambdas") required() action { (x,c) => c.copy(lambdas=x) }
    opt[Double]("gamma") required() action { (x,c) => c.copy(gamma=x) }
    opt[Long]("seed") required() action { (x,c) => c.copy(seed=x) }
    opt[String]("solver") required() action { (x,c) => c.copy(solver=x) }
  }.parse(args, YelpDCSolverConfig()).get

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

