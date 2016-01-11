package pipelines.images.cifar

import breeze.linalg._
import breeze.numerics._
import evaluation.MulticlassClassifierEvaluator
import loaders.GeneralCifarLoader
import nodes.images._
import nodes.learning.{BlockLeastSquaresEstimator, ZCAWhitener, ZCAWhitenerEstimator}
import nodes.stats.{StandardScaler, Sampler}
import nodes.util.{Cacher, ClassLabelIndicatorsFromIntLabels, MaxClassifier}
import pipelines.FunctionNode
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import pipelines.Logging
import scopt.OptionParser
import utils.{MatrixUtils, Stats, Image, ImageUtils}


class LabelAugmenter(mult: Int) extends FunctionNode[RDD[Int], RDD[Int]] {
  def apply(in: RDD[Int]) = in.flatMap(x => Seq.fill(mult)(x))
}

class ImageCrop(xStart: Int, yStart: Int,
                xEnd: Int, yEnd: Int) extends workflow.Transformer[Image, Image] {

  def apply(img: Image): Image = ImageUtils.crop(img, xStart, yStart, xEnd, yEnd)

}

object RandomPatchCifarFeaturizerWithAugmentation extends Serializable with Logging {
  val appName = "RandomPatchCifarFeaturizerWithAugmentation"

  def run(sc: SparkContext, conf: RandomCifarFeaturizerConfig) {
    //Set up some constants.
    val numClasses = 10
    val imageSize = conf.imageSize
    val numChannels = 3
    val whitenerSize = 100000

    val loader = new GeneralCifarLoader(imageSize, imageSize)

    // Load up training data, and optionally sample.
    val trainData = loader(sc, conf.trainLocation).cache()
    val trainImages = ImageExtractor(trainData)

    val patchExtractor = new Windower(conf.patchSteps, conf.patchSize)
      .andThen(ImageVectorizer.apply)
      .andThen(new Sampler(whitenerSize))

    val (filters, whitener): (DenseMatrix[Double], ZCAWhitener) = {
        val baseFilters = patchExtractor(trainImages)
        val baseFilterMat = Stats.normalizeRows(MatrixUtils.rowsToMatrix(baseFilters), 10.0)
        val whitener = new ZCAWhitenerEstimator(1e-1).fitSingle(baseFilterMat)

        //Normalize them.
        val sampleFilters = MatrixUtils.sampleRows(baseFilterMat, conf.numFilters)
        val unnormFilters = whitener(sampleFilters)
        val unnormSq = pow(unnormFilters, 2.0)
        val twoNorms = sqrt(sum(unnormSq(*, ::)))

        ((unnormFilters(::, *) / (twoNorms + 1e-10)) * whitener.whitener.t, whitener)
    }

    val convolver = new Convolver(filters, imageSize, imageSize, numChannels, Some(whitener), true)
    // TODO(stephentu): currently not flexible
    assert(convolver.resWidth == 27 && convolver.resHeight == 27)
    val stride = 2
    val windowSize = 19
    val nResulting = (0 until convolver.resWidth - windowSize + 1 by stride)
      .flatMap { _ => (0 until convolver.resHeight - windowSize + 1 by stride).map { _ => 1 } }
      .size

    println(s"convolver.resWidth ${convolver.resWidth} convolver.resHeight ${convolver.resHeight}")
    println(s"nResulting $nResulting")

    val pooling = SymmetricRectifier(alpha=conf.alpha)
        .andThen(new Pooler(conf.poolStride, conf.poolSize, identity, _.sum))
        .andThen(ImageVectorizer)
        .andThen(new Cacher[DenseVector[Double]])
    val trainFeaturesUnscaled: RDD[DenseVector[Double]] = pooling(new Windower(stride, windowSize).apply(convolver(trainImages)))
    val scaler = (new StandardScaler).fit(trainFeaturesUnscaled).andThen(new Cacher[DenseVector[Double]])

    val trainFeatures = scaler(trainFeaturesUnscaled)
    val trainLabels = new LabelAugmenter(nResulting).apply(LabelExtractor.apply(trainData)).cache()

    val testData = loader(sc, conf.testLocation)
    val testImages = ImageExtractor(testData)

    // gotta love spark

    trainLabels.zip(trainFeatures.map(_.toArray)).map { case (label, data) =>
      label + "," + data.mkString(",")
    }.saveAsTextFile(conf.trainOutfile)

    val testFeatures = convolver
        .andThen(new ImageCrop(4, 4, windowSize + 4, windowSize + 4))
        .andThen(pooling)
        .andThen(scaler)
        .apply(testImages)
    val testLabels = LabelExtractor.andThen(new Cacher[Int]).apply(testData)

    testLabels.zip(testFeatures.map(_.toArray)).map { case (label, data) =>
      label + "," + data.mkString(",")
    }.saveAsTextFile(conf.testOutfile)

  }

  case class RandomCifarFeaturizerConfig(
      trainLocation: String = "",
      testLocation: String = "",
      trainOutfile: String = "",
      testOutfile: String = "",
      numFilters: Int = 100,
      patchSize: Int = 6,
      patchSteps: Int = 1,
      poolSize: Int = 14,
      poolStride: Int = 13,
      alpha: Double = 0.25,
      imageSize: Int = 32)

  def parse(args: Array[String]): RandomCifarFeaturizerConfig = new OptionParser[RandomCifarFeaturizerConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[String]("trainOutfile") required() action { (x,c) => c.copy(trainOutfile=x) }
    opt[String]("testOutfile") required() action { (x,c) => c.copy(testOutfile=x) }
    opt[Int]("numFilters") action { (x,c) => c.copy(numFilters=x) }
    opt[Int]("patchSize") action { (x,c) => c.copy(patchSize=x) }
    opt[Int]("patchSteps") action { (x,c) => c.copy(patchSteps=x) }
    opt[Int]("poolSize") action { (x,c) => c.copy(poolSize=x) }
    opt[Double]("alpha") action { (x,c) => c.copy(alpha=x) }
    opt[Int]("imageSize") action { (x,c) => c.copy(imageSize=x) }
  }.parse(args, RandomCifarFeaturizerConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
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
