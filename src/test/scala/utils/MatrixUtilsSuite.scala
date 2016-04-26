package utils

import org.scalatest.FunSuite

import breeze.linalg._
import breeze.stats._

import org.apache.spark.SparkContext

import pipelines._

class MatrixUtilsSuite extends FunSuite with LocalSparkContext {

  test("computeMean works correctly") {
    val numRows = 1000
    val numCols = 32
    val numParts = 4
    sc = new SparkContext("local", "test")
    val in = DenseMatrix.rand(numRows, numCols)
    val inArr = MatrixUtils.matrixToRowArray(in)
    val rdd = sc.parallelize(inArr, numParts).mapPartitions { iter =>
      Iterator.single(MatrixUtils.rowsToMatrix(iter))
    }
    val expected = mean(in(::, *)).toDenseVector
    val actual = MatrixUtils.computeMean(rdd)
    assert(Stats.aboutEq(expected, actual, 1e-6))
  }

  test("squaredPDist works correctly") {
    val numRows = 75
    val numCols = 32
    val in = DenseMatrix.rand(numRows, numCols)
    val p = MatrixUtils.squaredPDist(in)

    var i = 0
    while (i < numRows) {
      var j = 0
      while (j < numRows) {
        val dist = norm(in(i, ::).t - in(j, ::).t)
        assert(Stats.aboutEq(dist * dist, p(i, j), 1e-6))
        j += 1
      }
      i += 1
    }

  }

}
