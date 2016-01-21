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


//class LabelAugmenter(mult: Int) extends FunctionNode[RDD[Int], RDD[Int]] {
//  def apply(in: RDD[Int]) = in.flatMap(x => Seq.fill(mult)(x))
//}
//
//class ImageCrop(xStart: Int, yStart: Int,
//                xEnd: Int, yEnd: Int) extends workflow.Transformer[Image, Image] {
//
//  def apply(img: Image): Image = ImageUtils.crop(img, xStart, yStart, xEnd, yEnd)
//
//}

object RandomPatchCifarFeaturizerRawAugment extends Serializable with Logging {
  val appName = "RandomPatchCifarFeaturizerRawAugment"

  def run(sc: SparkContext, conf: RandomCifarFeaturizerConfig) {
    //Set up some constants.
    val numClasses = 10
    val imageSize = conf.imageSize
    val numChannels = 3
    val whitenerSize = 100000
    val numRandomPatchesAugment = conf.numRandomPatchesAugment
    val augmentRandomPatchSize = 24

    val loader = new GeneralCifarLoader(imageSize, imageSize)

    // Load up training data, and optionally sample.
    val trainData = loader(sc, conf.trainLocation).cache()
    //Augment data here
    // flatmap operation on image. Just implement the algorithm.
    // use the Keystone Image class _. images utils.scala. don't use the windower. but look at the windower
    val randomFlipper = new RandomFlips(numRandomPatchesAugment, augmentRandomPatchSize)

    val trainImages = ImageExtractor.andThen(new Cacher[Image](Some("trainImages"))).apply(trainData)
    val trainImageIds = trainImages.zipWithIndex.map(x => x._2.toInt)
    val trainImageIdsAugmented = new LabelAugmenter(numRandomPatchesAugment).apply(trainImageIds)
    val trainImagesAugmented = randomFlipper(trainImages)

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

    val unscaledFeaturizer = new Convolver(filters, augmentRandomPatchSize, augmentRandomPatchSize, numChannels, Some(whitener), true)
        .andThen(SymmetricRectifier(alpha=conf.alpha))
        .andThen(new Pooler(conf.poolStride, conf.poolSize, identity, _.sum))
        .andThen(ImageVectorizer)
        .andThen(new Cacher[DenseVector[Double]])

    val featurizer = unscaledFeaturizer.andThen(new StandardScaler, trainImages)
        .andThen(new Cacher[DenseVector[Double]])

    val labelExtractor = LabelExtractor andThen new Cacher[Int]

    val trainFeatures = featurizer(trainImagesAugmented)
    val trainLabels = new LabelAugmenter(numRandomPatchesAugment).apply(labelExtractor(trainData)).cache()

    // gotta love spark
    trainLabels.zip(trainImageIdsAugmented).zip(trainFeatures.map(_.toArray)).map { case (labelIdx, data) =>
      labelIdx._2 + ".jpg," + labelIdx._1 + "," + data.mkString(",")
    }.saveAsTextFile(conf.trainOutfile)

    val testData = loader(sc, conf.testLocation)
    val testImages = ImageExtractor(testData)
    val testImageIds = testImages.zipWithIndex.map(x => x._2.toInt)

    val numTestAugment = 10 // 4 corners, center and flips of each of the 5

    val testImageIdsAugmented = new LabelAugmenter(numTestAugment).apply(testImageIds)
    val testImagesAugmented = new RandomFlips(numRandomPatchesAugment, augmentRandomPatchSize, centerCorners=true).apply(testImages)

    val testFeatures = featurizer(testImagesAugmented)
    val testLabels = new LabelAugmenter(numTestAugment).apply(labelExtractor(testData))

    testLabels.zip(testImageIdsAugmented).zip(testFeatures.map(_.toArray)).map { case (labelIdx, data) =>
      labelIdx._2 + ".jpg," + labelIdx._1 + "," + data.mkString(",")
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
      poolSize: Int = 10,
      poolStride: Int = 9,
      alpha: Double = 0.25,
      numRandomPatchesAugment: Int = 10,
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
    opt[Int]("numRandomPatchesAugment") action { (x,c) => c.copy(numRandomPatchesAugment=x) }
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
