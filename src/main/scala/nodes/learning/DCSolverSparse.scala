package nodes.learning

// vim: set ts=4 sw=4 et:

import org.netlib.util.intW
import com.github.fommil.netlib.BLAS.{getInstance=>blas}
import com.github.fommil.netlib.LAPACK.{getInstance=>lapack}

import breeze.linalg._
import breeze.numerics._

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import evaluation.{AugmentedExamplesEvaluator, MulticlassClassifierEvaluator, MulticlassMetrics}
import nodes.stats.SparseLinearKernel
import nodes.stats.UniformRangePartitioner
import nodes.util._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.SparkContext._
import pipelines._
import utils.{Image, MatrixUtils}
import workflow.Pipeline

import edu.berkeley.cs.amplab.mlmatrix.util.{Utils => MLMatrixUtils}

import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.ShuffledRDD

case class DCSolverYuchenSparseState(
  lambdas: Seq[Double],
  models: RDD[(Int, (Array[(Array[Long], Array[Double])], Seq[DenseMatrix[Double]]))] /* it is assumed each entry in the RDD belongs to a partition */
) extends Logging {

  def metrics(test: RDD[(Int, (Array[Long], Array[Double]))],
              numClasses: Int): Seq[MulticlassMetrics] = {

    val nModels = models.count()
    assert(nModels == models.partitions.size)

    var testAccums = test.mapPartitions { partition =>
      if (partition.hasNext) {
        val sz = partition.size
        assert(sz > 0)
        Iterator.single(lambdas.map { _ => DenseMatrix.zeros[Double](sz, numClasses) })
      } else {
        Iterator.empty
      }
    }.cache()
    testAccums.count()

    (0 until nModels.toInt).foreach { modelId =>
      // not sure of a better way to do this
      val (_, model) = models.filter { case (curModelId, _) => curModelId == modelId }.collect().head
      val modelBC = test.context.broadcast(model)

      val newTestAccums = testAccums.zipPartitions(test) { case (accumsIter, partition) =>
        if (accumsIter.hasNext) {
          val accums = accumsIter.next()
          val partitionSeq = partition.toArray

          val Xtest = partitionSeq.map(_._2) // Seq[(Array[Long], Array[Double])]
          val Ytest = partitionSeq.map(_._1) // Seq[Int]

          val (xTrainPart, alphaStarsPart) = modelBC.value
          val KtesttrainPart = new SparseLinearKernel().apply(Xtest, xTrainPart)

          accums.zip(alphaStarsPart).foreach { case (accum, alphaStarPart) =>
            accum += KtesttrainPart * alphaStarPart
          }

          Iterator.single(accums)
        } else {
          Iterator.empty
        }
      }.cache()
      newTestAccums.count()
      testAccums.unpersist()
      testAccums = newTestAccums

      //modelBC.destroy() // LEAK FOR NOW
      modelBC.unpersist()
      // TODO: truncate this lineage?
    }

    (0 until lambdas.size).map { idx =>
      MulticlassClassifierEvaluator(
        testAccums.map { evaluations: Seq[DenseMatrix[Double]] =>
          assert(lambdas.size == evaluations.size)
          val thisEvaluations = evaluations(idx)
          assert(thisEvaluations.cols == numClasses)
          MatrixUtils.matrixToRowArray(evaluations(idx) * (1.0 / nModels.toDouble)).map(MaxClassifier.apply).toSeq
        }.flatMap(x => x),
        test.map(x => x._1),
        numClasses)
    }
  }
}

object DCSolverYuchenSparse extends Logging {

  def isSorted[T](s: Seq[T])(implicit cmp: Ordering[T]): Boolean = {
    if (s.isEmpty) {
      true 
    } else {
      var i = 1
      while (i < s.size) {
        if (cmp.gt(s(i - 1), s(i)))
          return false
        i += 1
      }
      true
    }
  }

  private def addLambdaEyeInPlace(in: DenseMatrix[Double], lambda: Double): Unit = {
    assert(in.rows == in.cols)
    var i = 0
    while (i < in.rows) {
      in(i, i) += lambda
      i += 1
    }
  }

  private def subLambdaEyeInPlace(in: DenseMatrix[Double], lambda: Double): Unit = {
    assert(in.rows == in.cols)
    var i = 0
    while (i < in.rows) {
      in(i, i) -= lambda
      i += 1
    }
  }

  def fit(train: RDD[(Int, (Array[Long], Array[Double]))],
          numClasses: Int,
          lambdas: Seq[Double],
          numPartitions: Int,
          permutationSeed: Long): DCSolverYuchenSparseState = {
    val trainingStartTime = System.nanoTime()
    val sc = train.context

    val rng = new util.Random(permutationSeed)
    val numExamples = train.count().toInt
    val pi: Array[Int] = rng.shuffle((0 until numExamples).toIndexedSeq).toArray
    val piBC = sc.broadcast(pi)

    val shuffledTrainRnd = train
      .zipWithIndex
      .map { case (elem, idx) => (piBC.value.apply(idx.toInt), elem) }

    val part = new UniformRangePartitioner(numExamples, numPartitions)
    val ordering = Ordering[Int]
    val shuffledTrainSorted = new ShuffledRDD[Int, (Int, (Array[Long], Array[Double])), (Int, (Array[Long], Array[Double]))](
      shuffledTrainRnd, part).setKeyOrdering(ordering)
    val keysInOrder = shuffledTrainSorted.mapPartitions { iter => Iterator.single(isSorted(iter.map(_._1).toSeq)) }.collect() 
    println("TIMIT DCSolverYuchen, keysInOrder " + keysInOrder.forall(x => x))
    val shuffledTrain = shuffledTrainSorted
      .map(_._2)
      .cache()

    val shuffledPartCounts = shuffledTrain.mapPartitions(iter => Iterator.single(iter.size)).collect().mkString(",")
    println("TIMIT DCSolverYuchen, counts per part = " + shuffledPartCounts)

    piBC.unpersist()
    // piBC.destroy()

    val models = shuffledTrain.mapPartitionsWithIndex { case (partId, partition) =>
      val partitionSeq = partition.toArray
      val classLabeler = ClassLabelIndicatorsFromIntLabels(numClasses)
      val Xtrain = partitionSeq.map(_._2) //MatrixUtils.rowsToMatrix(partitionSeq.map(_._2))
      val Ytrain = MatrixUtils.rowsToMatrix(partitionSeq.map(_._1).map(classLabeler.apply))

      val localKernelStartTime = System.nanoTime()
      val Ktrain = new SparseLinearKernel().apply(Xtrain)
      println(s"PARTID_${partId}_KERNEL_GEN_TIME_${(System.nanoTime() - localKernelStartTime)/1e9}")

      if (lambdas.length == 1) {
        // Special case 1 lambda to be more memory efficient
        val localSolveStartTime = System.nanoTime()
        addLambdaEyeInPlace(Ktrain, lambdas(0))
        // val alphaStar = Ktrain \ Ytrain
        // square: LUSolve
        val alphaStar = DenseMatrix.zeros[Double](Ytrain.rows, Ytrain.cols)
        // we initialize alphaStar to Ytrain ??
        alphaStar := Ytrain
        val piv = new Array[Int](Ktrain.rows)
        // NOTE: we don't copy Ktrain, so it gets overwritten
        assert(!Ktrain.isTranspose)
        val info: Int = {
          val info = new intW(0)
          lapack.dgesv(Ktrain.rows, alphaStar.cols, Ktrain.data, Ktrain.offset, Ktrain.majorStride, piv, 0,
                       alphaStar.data, alphaStar.offset, alphaStar.majorStride, info)
          info.`val`
        }
       
        if (info > 0)
          throw new MatrixSingularException()
        else if (info < 0)
          throw new IllegalArgumentException()

        println(s"PARTID_${partId}_LAMBDA_${lambdas(0)}_LOCAL_SOLVE_TIME_${(System.nanoTime() - localSolveStartTime)/1e9} s")
        val trainLabels = partitionSeq.map(_._1)
        Iterator.single((partId, (Xtrain, Seq(alphaStar))))
      } else {
        val alphaStars = lambdas.map { lambda =>
          val localSolveStartTime = System.nanoTime()
          addLambdaEyeInPlace(Ktrain, lambda)
          val alphaStar = Ktrain \ Ytrain
          println(s"PARTID_${partId}_LAMBDA_${lambda}_LOCAL_SOLVE_TIME_${(System.nanoTime() - localSolveStartTime)/1e9} s")
          subLambdaEyeInPlace(Ktrain, lambda)

          val predictions = MatrixUtils.matrixToRowArray(Ktrain * alphaStar).map(MaxClassifier.apply).toSeq
          val trainLabels = partitionSeq.map(_._1)
          assert(predictions.size == trainLabels.size)

          val nErrors = predictions.zip(trainLabels).filter { case (x, y) => x != y }.size
          val trainErrorRate = (nErrors.toDouble / Xtrain.size.toDouble)
          val trainAccRate = 1.0 - trainErrorRate
          println(s"PARTID_${partId}_LAMBDA_${lambda}_TRAIN_ACC_${trainAccRate}")

          alphaStar
        }
        Iterator.single((partId, (Xtrain, alphaStars)))
      }
    }.cache()
    models.count()
    val totalTrainTime = (System.nanoTime() - trainingStartTime)/1e9
    println(s"TRAIN_TIME_${totalTrainTime}") 
    shuffledTrain.unpersist()

    DCSolverYuchenSparseState(lambdas, models)
  }

}

