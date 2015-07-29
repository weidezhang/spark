package org.apache.spark.ps

import org.apache.spark.SparkContext
import org.apache.spark.ps.local.LocalPSConfig
import org.scalatest.FunSuite

class UpdateSuite extends FunSuite with Serializable {
  private val sc: SparkContext = new SparkContext("local[2]", "test")
  test ("update") {
    UpdateTest(sc, 2, 1000000, 10)
  }

}

object UpdateTest {
  def apply(sc: SparkContext, numPartitions: Int, numFeatures: Int, numIterations: Int): Unit = {
    val psContext = new PSContext(sc, LocalPSConfig(1, numFeatures, 1))
    psContext.start()
    val initialParams = Array.fill[Double](numFeatures)(0.0)
    val data = sc.parallelize(1 to numPartitions, numPartitions)
    psContext.uploadParams(Array(initialParams))
    val time = System.nanoTime()
    psContext.runPSJob(data)((index, arr, client) => {
      for (i <- 0 until numIterations) {
        // get weights from parameter server
        val w = client.get(0)
        // sum of delta of current partition
        val delta = Array.fill[Double](numFeatures)(0.1)
        // update delta to parameter server
        client.update(0, delta)
        // end of current iteration
        client.clock()
      }
      Iterator()
    }).count()
    println("Avg iteration time:" + (System.nanoTime() - time) / 1e9 / numIterations)
    // download weights from parameter server
    val weights = psContext.downloadParams()(0)
    println(weights(0))
    // stop parameter server context
    psContext.stop()

  }
}
