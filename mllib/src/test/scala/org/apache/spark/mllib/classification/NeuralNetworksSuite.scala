package org.apache.spark.mllib.classification

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.{MLUtils, LocalSparkContext}
import org.scalatest.FunSuite
import scala.math.random

class NeuralNetworksSuite extends FunSuite with LocalSparkContext{

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


  test("model") {
    val data = inputs.zip(outputs).map{ case(features, label) =>
      new LabeledPoint(label, Vectors.dense(features))}
    val rddData = sc.parallelize(data, 2)
    val predictor = NeuralNetwork.train(rddData, Array(hiddenSize))
    val result = rddData.map(lp => (predictor.predict(lp.features), lp.label))
    result.foreach(x => println(x._1 +":" + x._2))
  }

  test("xor") {
    val sizes = Array(inputSize, hiddenSize, outputSize)
    var size = 0
    for(i <- 1 until sizes.size){
      size += sizes(i - 1) * sizes(i)
    }
    val nn = new NeuralNetworkGradient(sizes)
    /* training */
    var weights = Array.fill(size){random * (2.4 *2)- 2.4}
    for(i <- 0 until 100){
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

    println(Vectors.dense(weights))

  }

 test("real data") {
    val data = MLUtils.loadLibSVMFile(sc, "c:/ulanov/res/sentiment/thumbsup/films100.libsvm")
    val split = data.randomSplit(Array(0.9, 0.1), 11L)
    val training = split(0)
    val test = split(1)

   val predictor = NeuralNetwork.train(training, Array(60), 300)
   val predictionAndLabels = test.map { point =>
     val score = predictor.predict(point.features)
     (score, point.label)
   }
   val metrics = new MulticlassMetrics(predictionAndLabels)
   metrics.labels.foreach( l => println(metrics.fMeasure(l)))

  }


}
