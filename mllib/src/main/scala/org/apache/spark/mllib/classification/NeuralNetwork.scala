/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.classification

import scala.math._
import breeze.linalg.{axpy => brzAxpy, Vector => BV, DenseVector => BDV, DenseMatrix => BDM, sum => Bsum, norm => Bnorm}
import breeze.numerics.{sigmoid => Bsigmoid, round => Bround}
import breeze.generic._
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization.{SimpleUpdater, Updater, GradientDescent, Gradient}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.BLAS.{axpy}
import org.apache.spark.annotation.Experimental

/**
 * ::Experimental::
 * Trait for label conversion
 * for Neural Network
 */
@Experimental
private[classification] trait LabelConverter {

  /**
   * Returns the number of network outputs
   * (of the outer layer)
   */
  protected def resultCount: Int

  /**
   * Returns a vector of double that has 1.0
   * at the index equals to label.toInt and
   * other element are zero
   * When resultCount is 2 then
   * returns a vector of size 1 with label
   * @param label label
   */
  protected def label2Vector(label: Double): BDV[Double] = {
    val result = BDV.zeros[Double](resultCount)
    if (resultCount == 1) result(0) =
      (if ( label == -1) 0 else label) else result(label.toInt) = 1.0
    result
  }

  /**
   * Returns a label that corresponds to the
   * array of double
   * @param resultVector vector that represents a label for neural network
   */
  protected def vector2Label(resultVector: BDV[Double]): Double = {
    val resultArray = resultVector.toArray.map{ x => math.round(x)}
    if(resultArray.size == 1) resultArray(0) else resultArray.indexOf(1.0).toDouble
  }
}

private class NeuralNetworkGradient(val layers: Array[Int])
  extends Gradient with LabelConverter with Neural {

  override def resultCount = layers.last

  override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector):
    (linalg.Vector, Double) = {

    /* NB! weightMarices, gradientMatrices, errors have NULL zero element for addressing convenience */
    val (weightMatrices, bias) = unrollWeights(weights)
    /* forward run */
    val outputs = forwardRun(data, weightMatrices, bias)

    /* error back propagation */
    val errors = new Array[BDV[Double]](layers.size)
    val targetVector = label2Vector(label)
    for(i <- (layers.size - 1) until (0, -1)){
      val onesVector = BDV.ones[Double](outputs(i).length)
      val outPrime = (onesVector :- outputs(i)) :* outputs(i)
      if(i == layers.size - 1){
        errors(i) = (outputs(i) :- targetVector) :* outPrime
      }else{
        errors(i) = (weightMatrices(i + 1).t * errors(i + 1)) :* outPrime
      }
    }

    /* gradient */
    val gradientMatrices = new Array[BDM[Double]](layers.size)
    for(i <- (layers.size - 1) until (0, -1)) {
      gradientMatrices(i) = errors(i) * outputs(i - 1).t
    }
    val weightsGradient = rollWeights(gradientMatrices, errors)

    /* error */
    val delta = targetVector :- outputs(layers.size - 1)
    val outerError = Bsum(delta :* delta)

    (Vectors.fromBreeze(weightsGradient), outerError)
  }

  override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector,
                       cumGradient: linalg.Vector): Double = {
    val (gradient, loss) = compute(data, label, weights)
    axpy(1, gradient, cumGradient)
    loss
  }

  def initialWeights = Vectors.dense(Array.fill(weightCount){random * (2.4 * 2) - 2.4})
}

private class GradientUpdater extends Updater {
  override def compute(weightsOld: linalg.Vector, gradient: linalg.Vector,
                       stepSize: Double, iter: Int, regParam: Double): (linalg.Vector, Double) = {
    //println("Step:" + iter + " norm:" + Bnorm(gradient.toBreeze.toDenseVector))
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-stepSize, gradient.toBreeze, brzWeights)
    (Vectors.fromBreeze(brzWeights), 0)
  }
}

trait Neural {
  protected val layers: Array[Int]
  val weightCount =
    (for(i <- 1 until layers.size) yield (layers(i) * layers(i - 1))).sum +
      layers.sum - layers(0)

  def unrollWeights(weights: linalg.Vector): (Array[BDM[Double]], Array[BDV[Double]]) = {
    require(weights.size == weightCount)
    val weightsCopy = weights.toArray
    val weightMatrices = new Array[BDM[Double]](layers.size)
    var offset = 0
    for(i <- 1 until layers.size){
      weightMatrices(i) = new BDM[Double](layers(i), layers(i - 1), weightsCopy, offset)
      offset += layers(i) * layers(i - 1)
    }
    val bias = new Array[BDV[Double]](layers.size)
    for(i <- 1 until layers.size){
      bias(i) = new BDV[Double](weightsCopy, offset, 1, layers(i))
      offset += layers(i)
    }
    (weightMatrices, bias)
  }

  def rollWeights(weightMatricesUpdate: Array[BDM[Double]], biasUpdate: Array[BDV[Double]]) = {
    var weightsUpdate = weightMatricesUpdate(1).toDenseVector
    for(i <- 2 until layers.size) {
      weightsUpdate = BDV.vertcat(weightsUpdate, weightMatricesUpdate(i).toDenseVector)
    }
    for(i <- 1 until layers.size){
      weightsUpdate = BDV.vertcat(weightsUpdate, biasUpdate(i))
    }
    weightsUpdate
  }

  def forwardRun(data: linalg.Vector, weightMatrices: Array[BDM[Double]], bias: Array[BDV[Double]]): Array[BDV[Double]] = {
    val outArray = new Array[BDV[Double]](layers.size)
    outArray(0) = data.toBreeze.toDenseVector
    for(i <- 1 until layers.size) {
      outArray(i) = weightMatrices(i) * outArray(i - 1) :+ bias(i)
      Bsigmoid.inPlace(outArray(i))
    }
    outArray
  }
}

/**
 * ::Experimental::
 * Neural network model.
 *
 * @param layers array of layer sizes,
 * first is the input size of the network,
 * last is the output size of the network.
 * @param weights vector of weights of
 * neurons inputs, should have the size
 * input*hidden(0) + hidden(0)*hidden(1)
 * + ... + hidden(N)*output
 */
@Experimental
class NeuralNetworkModel(val layers: Array[Int], val weights: linalg.Vector)
  extends ClassificationModel with LabelConverter with Neural with Serializable {

  override def resultCount = layers.last
  private val (weightArray, bias) = unrollWeights(weights)

  override def predict(testData: RDD[linalg.Vector]): RDD[Double] = {
    testData.map(predict(_))
  }

  override def predict(testData: linalg.Vector): Double = {
    val outArray = forwardRun(testData, weightArray, bias)
    vector2Label(outArray(layers.size - 1))
  }
}

/**
 * ::Experimental::
 * Trains Neural network classifier
 * NOTE: labels should represent the class index
 */
@Experimental
class NeuralNetwork private (hiddenLayers: Array[Int], numIterations: Int, learningRate: Double)
  extends Serializable {

  def run(data: RDD[LabeledPoint]) = {
    val labels = data.map( lp => lp.label).distinct().collect()
    /* TODO: use LabelConverter instead! */
    val outputCount = if(labels.size == 2) 1 else labels.size
    val featureCount: Int = data.first().features.size
    val hl = if (hiddenLayers.size == 0) Array((featureCount + outputCount) / 2) else hiddenLayers
    val layers: Array[Int] = featureCount +: hl :+ outputCount
    val gradient = new NeuralNetworkGradient(layers)
    val updater = new GradientUpdater()
    val gradientDescent = new GradientDescent(gradient, updater)
    gradientDescent.setNumIterations(numIterations).setStepSize(learningRate)
    val tupleData = data.map(lp => (lp.label, lp.features))
    val weights = gradientDescent.optimize(tupleData, gradient.initialWeights)
    new NeuralNetworkModel(layers, weights)
  }
}

/**
 * ::Experimental::
 * Fabric for Neural Network classifier
 */
@Experimental
object NeuralNetwork {

  /**
   * Trains a Neural Network model given an RDD of `(label, features)` pairs.
   *
   * @param data RDD of `(label, array of features)` pairs.
   * @param hiddenLayers array of hidden layers sizes
   * @param numIterations number of iterations
   * @param learningRate learning rate
   */
  def train(data: RDD[LabeledPoint], hiddenLayers: Array[Int],
            numIterations: Int, learningRate: Double) : NeuralNetworkModel = {
    val nn = new NeuralNetwork(hiddenLayers, numIterations, learningRate)
    nn.run(data)
  }

  /**
   * Trains a Neural Network model given an RDD of `(label, features)` pairs.
   * Does 1000 iterations with learning rate 0.3
   *
   * @param data RDD of `(label, array of features)` pairs.
   * @param hiddenLayers array of hidden layers sizes
   */
  def train(data: RDD[LabeledPoint], hiddenLayers: Array[Int]) : NeuralNetworkModel = {
    train(data, hiddenLayers, 1000, 0.3)
  }

  /**
   * Trains a Neural Network model given an RDD of `(label, features)` pairs.
   * Creates one hidden layer of size = (input + ouput) / 2
   * Does 1000 iterations with learning rate 0.3
   *
   * @param data RDD of `(label, array of features)` pairs.
   */
  def train(data: RDD[LabeledPoint]) : NeuralNetworkModel = {
    train(data, Array(0), 1000, 0.3)
  }
}
