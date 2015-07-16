package org.apache.spark.ml.regression

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.ann.{FeedForwardTrainer, FeedForwardModel, FeedForwardTopology}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext

class MultilayerPerceptronRegressorSuite extends SparkFunSuite with MLlibTestSparkContext {

  test("XOR") {
    val inputs = Array[Array[Double]](
      Array[Double](0, 0),
      Array[Double](0, 1),
      Array[Double](1, 0),
      Array[Double](1, 1)
    )
    val outputs = Array[Double](0, 1, 1, 0)
    val data = inputs.zip(outputs).map { case (features, label) =>
      (Vectors.dense(features), Vectors.dense(Array(label)))
    }
    val rddData = sc.parallelize(data, 1)
    val dataFrame = sqlContext.createDataFrame(rddData)
    val hiddenLayersTopology = Array[Int](5)
    val dataSample = rddData.first()
    val layerSizes = dataSample._1.size +: hiddenLayersTopology :+ dataSample._2.size
    val trainer = new MultilayerPerceptronRegressor("mlpr")
//    val model = trainer.fit(dataFrame)
//    model.transform(dataFrame)
    // TODO: make predictions and labels comparison
  }

}
