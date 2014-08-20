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

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.Gradient
import org.apache.spark.rdd.RDD

import scala.math._

private[classification] trait NNetworkHelper {

  def layers: Array[Int]

  def sigmoid ( value: Double) : Double = 1d / (1d + exp(-value))

  lazy val weightCount = (for(i <- 1 until layers.size) yield layers(i - 1) * layers(i)).sum

  lazy val outputCount = layers.sum - (if (layers.size > 0) layers(0) else 0)

  def weightIndex(layer: Int, neuron: Int, connection: Int): Int = {
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

  def outputIndex(layer: Int, neuron: Int): Int = {
    var layerOffset = 0
    for (i <- 1 until layer) {
      layerOffset += layers(i)
    }
    layerOffset + neuron
  }

}

private[classification] trait LabelConverter {

  protected def resultCount: Int

  protected def label2Array(label: Double): Array[Double] = {
    val result = Array.fill(resultCount)(0.0)
    if (resultCount == 1) result(0) = label else result(label.toInt) = 1.0
    result
  }

  protected def array2Label(resultArray: Array[Int]): Double = {
    if(resultArray.size == 1) resultArray(0) else resultArray.indexOf(1.0).toDouble
  }
}


class NeuralNetworkGradient(layerSizes: Array[Int])
  extends Gradient with NNetworkHelper with LabelConverter{

  override def layers = layerSizes
  override def resultCount = layers.last
  var outputs = Array.fill(outputCount)(0.0)
  var errors = Array.fill(outputCount)(0.0)
  var gradient = Array.fill(weightCount)(0.0)

  override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector):
  (linalg.Vector, Double) = {
    /* Perceptron run */
    //The inputs for the first layer are the perceptron inputs
    val inputs: IndexedSeq[Double] = data.toArray
    /* TODO: outputs - hack */
    val targetOutputs = label2Array(label)
    val weightsCopy = weights.toArray
    /* loop through layers */
    for(layerIndex <- 1 until layerSizes.size) {
      /* loop through neurons in the layer */
      for(neuronIndex <- 0 until layerSizes(layerIndex)){
        /* run through neuron */
        var cumul : Double = 0
        if(layerIndex == 1){
          for(inputIndex <- 0 until inputs.size){
            cumul += inputs(inputIndex) *
              weightsCopy(weightIndex(layerIndex, neuronIndex, inputIndex))
          }
        }else{
          for(k <- 0 until layerSizes(layerIndex - 1)){
            cumul += outputs(outputIndex(layerIndex - 1, k)) *
              weightsCopy(weightIndex(layerIndex, neuronIndex, k))
          }
        }
        /* TODO: add bias! */
        outputs(outputIndex(layerIndex, neuronIndex)) = sigmoid(cumul)
      }
    }
    /* perceptron error back propagation */
    //For each layer from last to first
    for(layerIndex <- layerSizes.size - 1 to (1, -1)) {
      //For every neuron in the layer
      for(neuronIndex <- 0 until layerSizes(layerIndex)){
        val neuronOutput = outputs(outputIndex(layerIndex, neuronIndex))
        //Last factor of the error calculation
        if(layerIndex == layerSizes.size - 1){ //Last layer
          errors(outputIndex(layerIndex, neuronIndex)) =
            neuronOutput * (1 - neuronOutput) *
              (targetOutputs(neuronIndex) - neuronOutput)
        } else { //Hidden layers
          var tmp : Double = 0
          for(nextNeuronIndex <- 0 until layerSizes(layerIndex + 1)) {
            tmp += errors(outputIndex(layerIndex + 1, nextNeuronIndex)) *
              weightsCopy( weightIndex(layerIndex + 1, nextNeuronIndex, neuronIndex))
          }
          errors(outputIndex(layerIndex, neuronIndex)) = neuronOutput * (1 - neuronOutput) * tmp
        }
      }
    }

    /* perceptron weights gradient computation */
    for(layerIndex <- layerSizes.size - 1 to (1, -1)) {
      val previousLayer = layerIndex - 1
      for(neuronIndex <- 0 until layerSizes(layerIndex)){
        val neuronError = errors(outputIndex(layerIndex, neuronIndex))
        for(previusNeuronIndex <- 0 until layerSizes(previousLayer)){
          gradient(weightIndex(layerIndex, neuronIndex, previusNeuronIndex)) =
            if(layerIndex == 1) {
              neuronError * inputs(previusNeuronIndex)/* why times weight doesn't work? */
            }else {
              neuronError * outputs(outputIndex(previousLayer, previusNeuronIndex))
            }
        }
      }
    }

    var cumul : Double = 0
    for (i <- 0 until targetOutputs.size) {
      cumul += pow(targetOutputs(i) - outputs(outputIndex(layerSizes.size - 1, i)), 2)
      println("output/target:" + outputs(outputIndex(layerSizes.size - 1, i)) +
        "/" + targetOutputs(i))
    }
    sqrt(cumul)

    (Vectors.dense(gradient.clone()), cumul)
  }

  override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector,
                       cumGradient: linalg.Vector): Double = {
    1.0
  }

  def initialWeights = 0
}

class NeuralNetworkModel(layerArray: Array[Int], val weights: linalg.Vector)
  extends ClassificationModel with NNetworkHelper with LabelConverter with Serializable {

  override def layers = layerArray
  override def resultCount = layers.last

  require(weightCount == weights.size)

  override def predict(testData: RDD[linalg.Vector]): RDD[Double] = {
    testData.map(predict(_))
  }

  override def predict(testData: linalg.Vector): Double = {
    val inputs = testData.toArray
    val outputs = Array.fill(outputCount){0.0}
    for(layerIndex <- 1 until layers.size) {
      for(neuronIndex <- 0 until layers(layerIndex)){
        var cumul : Double = 0
        if(layerIndex == 1){
          for(inputIndex <- 0 until inputs.size){
            cumul += inputs(inputIndex) *
              weights(weightIndex(layerIndex, neuronIndex, inputIndex))
          }
        }else{
          for(k <- 0 until layers(layerIndex - 1)){
            cumul += outputs(outputIndex(layerIndex - 1, k)) *
              weights(weightIndex(layerIndex, neuronIndex, k))
          }
        }
        /* TODO: add bias! */
        outputs(outputIndex(layerIndex, neuronIndex)) = sigmoid(cumul)
      }
    }
    val lastLayer = layers.size - 1
    val lastLayerSize = layers.last
    val result = outputs.slice(outputIndex(lastLayer, 0), outputIndex(lastLayer, lastLayerSize))
      .map(math.round(_).toInt)
    array2Label(result)
  }
}

object NeuralNetworks {

}
