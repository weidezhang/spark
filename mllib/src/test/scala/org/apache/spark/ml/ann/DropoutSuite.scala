package org.apache.spark.ml.ann

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.classification.LabelConverter
import org.apache.spark.mllib.util.{MLUtils, MLlibTestSparkContext}

class DropoutSuite extends SparkFunSuite with MLlibTestSparkContext {

  // TODO: make a better test here
  test("dropout mnist") {
    val lpTrain = MLUtils.loadLibSVMFile(sc, "c:/ulanov/res/mnist/mnist.scale.t.780")
    val lpTest = MLUtils.loadLibSVMFile(sc, "c:/ulanov/res/mnist/mnist.scale.t.780")
    val layers = Array[Int](780, 10)
    val train = lpTrain.map(lp => LabelConverter.encodeLabeledPoint(lp, layers.last))

    val defaultTopology = FeedForwardTopology.multiLayerPerceptron(layers, true)
    val topology = new DropoutTopology(defaultTopology.layers, 0.9, 0.5)
    val trainer = new FeedForwardTrainer(topology, layers(0), layers.last)
    trainer.LBFGSOptimizer.setConvergenceTol(1e-4).setNumIterations(40)
    trainer.setStackSize(10)
    val model = trainer.train(train)
    val predictionAndLabels = lpTest.map( lp =>
      (LabelConverter.decodeLabel(model.predict(lp.features)), lp.label))
    val accuracy = predictionAndLabels.map{ case(p, l) => if (p == l) 1 else 0}.sum() / predictionAndLabels.count()
    println("Accuracy:" + accuracy)

  }

}
