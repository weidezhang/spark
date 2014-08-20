package org.apache.spark.mllib.classification

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.LocalSparkContext
import org.scalatest.FunSuite
import scala.math.random

class NeuralNetworksSuite extends FunSuite with LocalSparkContext{

  test("xor") {
    /* training set */
    val inputs = Array[Array[Double]](
      Array[Double](0,0),
      Array[Double](0,1),
      Array[Double](1,0),
      Array[Double](1,1)
    )
    val outputs = Array[Double](0, 1, 1, 0)
    /* NN */
    val inputSize = 2
    val hiddenSize = 5
    val outputSize = 1
    val sizes = Array(inputSize, hiddenSize, outputSize)
    var size = 0
    for(i <- 1 until sizes.size){
      size += sizes(i - 1) * sizes(i)
    }
    val nn = new NeuralNetworkGradient(sizes)
    /* training */
    var weights = Array.fill(size){random * (2.4 *2)- 2.4}
    for(i <- 0 until 1000){
      for(j <-0 until inputs.size){
        val (gradient, loss) = nn.compute(Vectors.dense(inputs(j)), outputs(j), Vectors.dense(weights))
        weights = weights.zip(gradient.toArray).map{ case(x, y) => x + y }
        println("loss:" + loss)
      }
    }
    val model = new NeuralNetworkModel(sizes, Vectors.dense(weights))
    for(j <- 0 until inputs.size){
      println(model.predict(Vectors.dense(inputs(j))))
    }


  }

}
