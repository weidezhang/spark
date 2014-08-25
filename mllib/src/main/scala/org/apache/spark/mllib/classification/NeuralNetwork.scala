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
import breeze.linalg.{axpy => brzAxpy, Vector => BV, DenseVector => BDV, DenseMatrix => BDM, sum => Bsum}
import breeze.numerics.{sigmoid => Bsigmoid, round => Bround}
import breeze.generic._
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{Updater, GradientDescent, Gradient}
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
    if (resultCount == 1) result(0) = label else result(label.toInt) = 1.0
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
  extends Gradient with LabelConverter{

  override def resultCount = layers.last

  private lazy val weightCount =
    (for(i <- 1 until layers.size) yield layers(i - 1) * layers(i)).sum


  override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector):
    (linalg.Vector, Double) = {

    /* NB! weightMarices, gradients, errors have NULL zero element for addressing convenience */
    val weightMarices = new Array[BDM[Double]](layers.size)
    var offset = 0
    val weightsCopy = weights.toArray
    for(i <- 1 until layers.size){
      weightMarices(i) = new BDM[Double](layers(i), layers(i - 1), weightsCopy, offset)
      offset += layers(i) * layers(i - 1)
    }

    /* neural network forward propagation */
    val outputs = new Array[BDV[Double]](layers.size)
    outputs(0) = data.toBreeze.toDenseVector
    for(i <- 1 until layers.size) {
      outputs(i) = weightMarices(i) * outputs(i - 1)
      Bsigmoid.inPlace(outputs(i))
    }

    /* error back propagation */
    val errors = new Array[BDV[Double]](layers.size)
    val targetVector = label2Vector(label)
    for(i <- (layers.size - 1) until (0, -1)){
      val onesVector = BDV.ones[Double](outputs(i).length)
      val outPrime = ( onesVector :- outputs(i)) :* outputs(i)
      if(i == layers.size - 1){
        errors(i) = (targetVector :- outputs(i)) :* outPrime
      }else{
        errors(i) = (weightMarices(i + 1).t * errors(i + 1)) :* outPrime
      }
    }

    /* gradient */
    val gradients = new Array[BDM[Double]](layers.size)
    for(i <- (layers.size - 1) until (0, -1)) {
      gradients(i) = errors(i) * outputs(i - 1).t
    }
    var gV = gradients(1).toDenseVector
    for(i <- 2 until layers.size) {
      gV = BDV.vertcat(gV, gradients(i).toDenseVector)
    }

    /*  breeze error */
    val delta = targetVector :- outputs(layers.size - 1)
    val outerError = Bsum(delta :* delta)
    (Vectors.fromBreeze(gV), outerError)
  }

  override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector,
                       cumGradient: linalg.Vector): Double = {
    /* TODO: try to remove gradient copying (compute returns clone) */
    val (gradient, loss) = compute(data, label, weights)
    axpy(1, gradient, cumGradient)
    loss
  }

  def initialWeights = Vectors.dense(Array.fill(weightCount){random * (2.4 *2)- 2.4})
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
  extends ClassificationModel with LabelConverter with Serializable {

  override def resultCount = layers.last
  //require(weightCount == weights.size)
  private val weightsCopy = weights.toArray
  private val weightArray = new Array[BDM[Double]](layers.size)
  var offset = 0
  for(i <- 1 until layers.size){
    weightArray(i) = new BDM[Double](layers(i), layers(i - 1), weightsCopy, offset)
    offset += layers(i) * layers(i - 1)
  }

  override def predict(testData: RDD[linalg.Vector]): RDD[Double] = {
    testData.map(predict(_))
  }

  override def predict(testData: linalg.Vector): Double = {
    /* TODO: share this code with Gradient forward run */
    val outArray = new Array[BDV[Double]](layers.size)
    outArray(0) = testData.toBreeze.toDenseVector
    for(i <- 1 until layers.size) {
      outArray(i) = weightArray(i) * outArray(i - 1)
      Bsigmoid.inPlace(outArray(i))
    }
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

private class GradientUpdater extends Updater {
  override def compute(weightsOld: linalg.Vector, gradient: linalg.Vector,
                       stepSize: Double, iter: Int, regParam: Double): (linalg.Vector, Double) = {
    //println("Step:" + iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(stepSize, gradient.toBreeze, brzWeights)
    (Vectors.fromBreeze(brzWeights), 0)
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
   * Does 1000 iterations with learning rate 0.9
   *
   * @param data RDD of `(label, array of features)` pairs.
   * @param hiddenLayers array of hidden layers sizes
   */
  def train(data: RDD[LabeledPoint], hiddenLayers: Array[Int]) : NeuralNetworkModel = {
    train(data, hiddenLayers, 1000, 0.9)
  }

  /**
   * Trains a Neural Network model given an RDD of `(label, features)` pairs.
   * Creates one hidden layer of size = (input + ouput) / 2
   * Does 1000 iterations with learning rate 0.9
   *
   * @param data RDD of `(label, array of features)` pairs.
   */
  def train(data: RDD[LabeledPoint]) : NeuralNetworkModel = {
    train(data, Array(0), 1000, 0.9)
  }
}
