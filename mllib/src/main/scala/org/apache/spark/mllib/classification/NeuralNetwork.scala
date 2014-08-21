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
import breeze.linalg.{axpy => brzAxpy, Vector => BV}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{Updater, GradientDescent, Gradient}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.BLAS.{axpy}
import org.apache.spark.annotation.Experimental

/**
 * ::Experimental::
 * Helper Trait for neural network
 * Implements addressing to array of weights
 */
@Experimental
private[classification] trait NeuralNetworkHelper {

  def layers: Array[Int]
  def sigmoid ( value: Double) : Double = 1d / (1d + exp(-value))
  protected lazy val weightCount =
    (for(i <- 1 until layers.size) yield layers(i - 1) * layers(i)).sum
  protected lazy val outputCount = layers.sum - (if (layers.size > 0) layers(0) else 0)

  protected def weightIndex(layer: Int, neuron: Int, connection: Int): Int = {
    var layerOffset = 0
    if(layer == 0) {
      layerOffset = 0
    } else {
      for (i <- 1 until layer) {
        layerOffset += layers(i - 1) * layers(i)
      }
    }
    layerOffset + neuron * layers(layer - 1) + connection
  }

  protected def outputIndex(layer: Int, neuron: Int): Int = {
    var layerOffset = 0
    for (i <- 1 until layer) {
      layerOffset += layers(i)
    }
    layerOffset + neuron
  }

  protected def forwardRun(weights: Array[Double], inputs: Array[Double]) = {
    /* TODO implement forward run and share across gradient and model */
  }
}

/**
 * ::Experimental::
 * Trait for label conversion
 * for Neural Network
 */
@Experimental
private[classification] trait LabelConverter {

  /**
   * Returns the number of network ouputs
   * (of the outer layer)
   */
  protected def resultCount: Int

  /**
   * Returns an array of double that has 1.0
   * at the index equals to label.toInt and
   * other element are zero
   * When resultCount is 2 then
   * returns array of size 1 with label
   * @param label label
   */
  protected def label2Array(label: Double): Array[Double] = {
    val result = Array.fill(resultCount)(0.0)
    if (resultCount == 1) result(0) = label else result(label.toInt) = 1.0
    result
  }

  /**
   * Returns a label that corresponds to the
   * array of double
   * @param resultArray array that represents a label for neural network
   */
  protected def array2Label(resultArray: Array[Int]): Double = {
    if(resultArray.size == 1) resultArray(0) else resultArray.indexOf(1.0).toDouble
  }
}

private class NeuralNetworkGradient(layerSizes: Array[Int])
  extends Gradient with NeuralNetworkHelper with LabelConverter{

  override def layers = layerSizes
  override def resultCount = layers.last
  val outputs = Array.fill(outputCount)(0.0)
  val errors = Array.fill(outputCount)(0.0)
  val gradient = Array.fill(weightCount)(0.0)

  override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector):
  (linalg.Vector, Double) = {
    /* Perceptron run */
    val inputs: IndexedSeq[Double] = data.toArray
    val targetOutputs = label2Array(label)
    val weightsCopy = weights.toArray
    var layerIndex = 1     /* loop through layers */
    while (layerIndex < layers.size) {
      var neuronIndex = 0  /* loop through neurons in the layer */
      while (neuronIndex < layers(layerIndex)){
        var cumul : Double = 0
        var inputIndex = 0 /* run through neuron */
        if(layerIndex == 1){
          while (inputIndex < inputs.size){
            cumul += inputs(inputIndex) *
              weightsCopy(weightIndex(layerIndex, neuronIndex, inputIndex))
            inputIndex += 1
          }
        }else{
          while (inputIndex < layers(layerIndex - 1)){
            cumul += outputs(outputIndex(layerIndex - 1, inputIndex)) *
              weightsCopy(weightIndex(layerIndex, neuronIndex, inputIndex))
            inputIndex += 1
          }
        }
        /* TODO: add bias! */
        outputs(outputIndex(layerIndex, neuronIndex)) = sigmoid(cumul)
        neuronIndex += 1
      }
      layerIndex += 1
    }
    /* perceptron error back propagation */
    layerIndex = layers.size - 1  //For each layer from last to first
    while (layerIndex > 0) {
      var neuronIndex = 0             //For every neuron in the layer
      while (neuronIndex < layers(layerIndex)){
        val neuronOutput = outputs(outputIndex(layerIndex, neuronIndex))
        if(layerIndex == layers.size - 1){ //Last layer
          errors(outputIndex(layerIndex, neuronIndex)) =
            neuronOutput * (1 - neuronOutput) *
              (targetOutputs(neuronIndex) - neuronOutput)
        } else { //Hidden layers
          var tmp : Double = 0
          var nextNeuronIndex = 0
          while (nextNeuronIndex < layers(layerIndex + 1)) {
            tmp += errors(outputIndex(layerIndex + 1, nextNeuronIndex)) *
              weightsCopy( weightIndex(layerIndex + 1, nextNeuronIndex, neuronIndex))
            nextNeuronIndex += 1
          }
          errors(outputIndex(layerIndex, neuronIndex)) = neuronOutput * (1 - neuronOutput) * tmp
        }
        neuronIndex += 1
      }
      layerIndex -= 1
    }
    /* perceptron weights gradient computation */
    layerIndex = layers.size - 1
    while (layerIndex > 0) {
      val previousLayer = layerIndex - 1
      var neuronIndex = 0
      while (neuronIndex < layers(layerIndex)){
        val neuronError = errors(outputIndex(layerIndex, neuronIndex))
        var previousNeuronIndex = 0
        while (previousNeuronIndex < layers(previousLayer)){
          gradient(weightIndex(layerIndex, neuronIndex, previousNeuronIndex)) =
            if(layerIndex == 1) {
              neuronError * inputs(previousNeuronIndex)/* why times weight doesn't work? */
            }else {
              neuronError * outputs(outputIndex(previousLayer, previousNeuronIndex))
            }
          previousNeuronIndex += 1
        }
        neuronIndex += 1
      }
      layerIndex -= 1
    }
    var error : Double = 0
    var targetIndex = 0
    while (targetIndex < targetOutputs.size) {
      error += pow(targetOutputs(targetIndex) -
        outputs(outputIndex(layers.size - 1, targetIndex)), 2)
      targetIndex += 1
    }
    (Vectors.dense(gradient.clone()), error)
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
 * @param layerArray array of layer sizes,
 * first is the input size of the network,
 * last is the output size of the network.
 * @param weights vector of weights of
 * neurons inputs, should have the size
 * input*hidden(0) + hidden(0)*hidden(1)
 * + ... + hidden(N)*output
 */
@Experimental
class NeuralNetworkModel(layerArray: Array[Int], val weights: linalg.Vector)
  extends ClassificationModel with NeuralNetworkHelper with LabelConverter with Serializable {

  override def layers = layerArray
  override def resultCount = layers.last
  require(weightCount == weights.size)
  private val outputs = Array.fill(outputCount){0.0}

  override def predict(testData: RDD[linalg.Vector]): RDD[Double] = {
    testData.map(predict(_))
  }

  override def predict(testData: linalg.Vector): Double = {
    val inputs = testData.toArray
    /* TODO: share this code with Gradient forward run */
    var layerIndex = 1     /* loop through layers */
    while (layerIndex < layers.size) {
      var neuronIndex = 0  /* loop through neurons in the layer */
      while (neuronIndex < layers(layerIndex)){
        var cumul : Double = 0
        var inputIndex = 0 /* run through neuron */
        if(layerIndex == 1){
          while (inputIndex < inputs.size){
            cumul += inputs(inputIndex) *
              weights(weightIndex(layerIndex, neuronIndex, inputIndex))
            inputIndex += 1
          }
        }else{
          while (inputIndex < layers(layerIndex - 1)){
            cumul += outputs(outputIndex(layerIndex - 1, inputIndex)) *
              weights(weightIndex(layerIndex, neuronIndex, inputIndex))
            inputIndex += 1
          }
        }
        /* TODO: add bias! */
        outputs(outputIndex(layerIndex, neuronIndex)) = sigmoid(cumul)
        neuronIndex += 1
      }
      layerIndex += 1
    }
    val lastLayer = layers.size - 1
    val lastLayerSize = layers.last
    val result = outputs.slice(outputIndex(lastLayer, 0), outputIndex(lastLayer, lastLayerSize))
      .map(math.round(_).toInt)
    array2Label(result)
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
