package pipelines.images.mnist

import breeze.linalg.DenseVector
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import evaluation.MulticlassClassifierEvaluator
import loaders.{CsvDataLoader, LabeledData}
import nodes.learning._
import nodes.stats.{LinearRectifier, PaddedFFT, RandomSignNode, NormalizeRows}
import nodes.util._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import pipelines._
import scopt.OptionParser
import utils.Image
import workflow.Pipeline
import workflow.LabelEstimator


object MnistRandomFFT extends Serializable with Logging {
  val appName = "MnistRandomFFT"

  def run(sc: SparkContext, conf: MnistRandomFFTConfig): Pipeline[DenseVector[Double], Int] = {
    // This is a property of the MNIST Dataset (digits 0 - 9)
    val numClasses = 10

    val randomSignSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(conf.seed)))

    // The number of pixels in an MNIST image (28 x 28 = 784)
    // Because the mnistImageSize is 784, we get 512 PaddedFFT features per FFT.
    val mnistImageSize = 784

    val startTime = System.nanoTime()

    val train = LabeledData(
      CsvDataLoader(sc, conf.trainLocation, conf.numPartitions)
        // The pipeline expects 0-indexed class labels, but the labels in the file are 1-indexed
        .map(x => (x(0).toInt - 1, x(1 until x.length)))
        .repartition(conf.numPartitions)
        .cache())
    val labels = ClassLabelIndicatorsFromIntLabels(numClasses).apply(train.labels)

    val featurizer = Pipeline.gather {
      Seq.fill(conf.numFFTs) {
        RandomSignNode(mnistImageSize, randomSignSource) andThen PaddedFFT() andThen LinearRectifier(0.0)
      }
    } andThen VectorCombiner() andThen NormalizeRows andThen (new Cacher[DenseVector[Double]])

    val solver: LabelEstimator[DenseVector[Double], DenseVector[Double], DenseVector[Double]] =
      if (conf.solver == "bcd") {
        new BlockLeastSquaresEstimator(conf.blockSize, 1, conf.lambda.getOrElse(0), computeCost=true)
      } else if (conf.solver == "lbfgs") {
        new BatchLBFGSwithL2(new LeastSquaresBatchGradient, numIterations=conf.numIters,
            regParam=conf.lambda.getOrElse(0))
      } else if (conf.solver == "sgd") {
        new MiniBatchSGDwithL2(new LeastSquaresBatchGradient, numIterations=conf.numIters,
            stepSize=conf.sgdStepSize, dampen=conf.sgdDampen,
            miniBatchFraction=conf.sgdMiniBatchFraction, regParam=conf.lambda.getOrElse(0))
      } else if (conf.solver == "cocoa") {
        new CocoaSDCAwithL2(new LeastSquaresBatchGradient, numIterations=conf.numIters,
          regParam=conf.lambda.getOrElse(0), beta=conf.cocoaBeta, computeCost=false,
          numLocalItersFraction=conf.cocoaNumLocalItersFraction)
      } else {
        new BlockLeastSquaresEstimator(conf.blockSize, 1, conf.lambda.getOrElse(0), computeCost=true)
      }

    val featurized = featurizer.apply(train.data)
    val model = solver.withData(featurized, labels)
    val pipeline = featurizer andThen model andThen MaxClassifier

    val test = LabeledData(
      CsvDataLoader(sc, conf.testLocation, conf.numPartitions)
        // The pipeline expects 0-indexed class labels, but the labels in the file are 1-indexed
        .map(x => (x(0).toInt - 1, x(1 until x.length)))
        .cache())

    // Calculate train error
    val trainEval = MulticlassClassifierEvaluator(pipeline(train.data), train.labels, numClasses)
    logInfo("TRAIN Error is " + (100 * trainEval.totalError) + "%")

    // Calculate test error
    val testEval = MulticlassClassifierEvaluator(pipeline(test.data), test.labels, numClasses)
    logInfo("TEST Error is " + (100 * testEval.totalError) + "%")

    val endTime = System.nanoTime()
    logInfo(s"Pipeline took ${(endTime - startTime)/1e9} s")

    pipeline
  }

  case class MnistRandomFFTConfig(
      trainLocation: String = "",
      testLocation: String = "",
      numFFTs: Int = 200,
      blockSize: Int = 2048,
      numPartitions: Int = 10,
      solver: String = "bcd",
      numIters: Int = 10,
      sgdStepSize: Double = 0.1,
      sgdDampen: Option[Double] = None,
      sgdMiniBatchFraction: Double = 1.0,
      cocoaNumLocalItersFraction: Double = 1.0,
      cocoaBeta: Double = 1.0,
      lambda: Option[Double] = None,
      seed: Long = 0)

  def parse(args: Array[String]): MnistRandomFFTConfig = new OptionParser[MnistRandomFFTConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[String]("solver") action { (x,c) => c.copy(solver=x) }
    opt[Int]("numIters") action { (x,c) => c.copy(numIters=x) }
    opt[Double]("sgdStepSize") action { (x,c) => c.copy(sgdStepSize=x) }
    opt[Double]("sgdDampen") action { (x,c) => c.copy(sgdDampen=Some(x)) }
    opt[Double]("sgdMiniBatchFraction") action { (x,c) => c.copy(sgdMiniBatchFraction=x) }
    opt[Double]("cocoaBeta") action { (x,c) => c.copy(cocoaBeta=x) }
    opt[Double]("cocoaNumLocalItersFraction") action { (x,c) => c.copy(cocoaNumLocalItersFraction=x) }
    opt[Int]("numFFTs") action { (x,c) => c.copy(numFFTs=x) }
    opt[Int]("blockSize") validate { x =>
      // Bitwise trick to test if x is a power of 2
      if (x % 512 == 0) {
        success
      } else  {
        failure("Option --blockSize must be divisible by 512")
      }
    } action { (x,c) => c.copy(blockSize=x) }
    opt[Int]("numPartitions") action { (x,c) => c.copy(numPartitions=x) }
    opt[Double]("lambda") action { (x,c) => c.copy(lambda=Some(x)) }
    opt[Long]("seed") action { (x,c) => c.copy(seed=x) }
  }.parse(args, MnistRandomFFTConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   *
   * @param args
   */
  def main(args: Array[String]) = {
    val appConfig = parse(args)

    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]")
    val sc = new SparkContext(conf)
    run(sc, appConfig)

    sc.stop()
  }
}
