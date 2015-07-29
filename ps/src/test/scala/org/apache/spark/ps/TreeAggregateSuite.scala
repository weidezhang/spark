package org.apache.spark.ps

import org.apache.spark.SparkContext
import org.scalatest.FunSuite

class TreeAggregateSuite extends FunSuite {

  private val sc: SparkContext = new SparkContext("local[2]", "test")

  test("treeAggregate update") {
    TreeAggregateTest(sc, 2, 1000000, 10)
  }

}

object TreeAggregateTest {

  def apply(sc: SparkContext, numPartitions: Int,  numFeatures: Int, numIterations: Int): Unit = {
    val data = sc.parallelize(1 to numPartitions, numPartitions)
    var weights = Array.fill[Double](numFeatures)(0.0)
    val time = System.nanoTime()
    for (i <- 0 until numIterations) {
      val bcWeights = data.context.broadcast(weights)
      val sum = data.treeAggregate[Array[Double]](Array.fill[Double](numFeatures)(0.0)) (
        seqOp = (c, v) => c.zip(Array.fill(numFeatures)(0.1)).map(t => t._1 + t._2),
        combOp = (c1, c2) => c1.zip(c2).map(t => t._1 + t._2)
      )
      weights = weights.zip(sum).map(t => t._1 + t._2)
    }
    println("Avg iteration time:" + (System.nanoTime() - time) / 1e9 / numIterations)
    println(weights(0))
  }
}