package evaluation

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

/**
 * RMSE Evaluator
 * from the encoding eval toolkit at http://www.robots.ox.ac.uk/~vgg/software/enceval_toolkit/
 */
object RMSEEvaluator {

  def apply(testActual: RDD[Int], testPredicted: RDD[Int]): Double = {
		val numTest = testActual.count
    val squareErr = testPredicted.zip(testActual).map { case (p, a) =>
      math.pow(p - a, 2.0)
    }.reduce(_ + _)
    
    val rmseCalc = math.sqrt(squareErr / numTest)
		rmseCalc
  }
}
