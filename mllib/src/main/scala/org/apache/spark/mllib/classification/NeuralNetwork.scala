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
import breeze.linalg.{axpy => brzAxpy, Vector => BV, DenseVector => BDV,
DenseMatrix => BDM, sum => Bsum, argmax => Bargmax, norm => Bnorm}
import breeze.numerics.{sigmoid => Bsigmoid}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization.{Updater, GradientDescent, Gradient}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.BLAS.{axpy}
import org.apache.spark.annotation.Experimental

/**
 * ::Experimental::
 * Trait for roll/unroll weights and forward/back propagation in neural network
 */
@Experimental
private[classification] trait NeuralHelper {
  protected val layers: Array[Int]
  protected val weightCount =
    (for(i <- 1 until layers.size) yield (layers(i) * layers(i - 1))).sum +
      layers.sum - layers(0)

  protected def unrollWeights(weights: linalg.Vector): (Array[BDM[Double]], Array[BDV[Double]]) = {
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

  protected def rollWeights(weightMatricesUpdate: Array[BDM[Double]], biasUpdate: Array[BDV[Double]]) = {
    var weightsUpdate = weightMatricesUpdate(1).toDenseVector
    for(i <- 2 until layers.size) {
      weightsUpdate = BDV.vertcat(weightsUpdate, weightMatricesUpdate(i).toDenseVector)
    }
    for(i <- 1 until layers.size){
      weightsUpdate = BDV.vertcat(weightsUpdate, biasUpdate(i))
    }
    weightsUpdate
  }

  protected def forwardRun(data: BDV[Double], weightMatrices: Array[BDM[Double]],
                           bias: Array[BDV[Double]]): Array[BDV[Double]] = {
    val outArray = new Array[BDV[Double]](layers.size)
    outArray(0) = data
    for(i <- 1 until layers.size) {
      outArray(i) = weightMatrices(i) * outArray(i - 1) :+ bias(i)
      Bsigmoid.inPlace(outArray(i))
    }
    outArray
  }

  protected def wGradient(weightMatrices: Array[BDM[Double]],
                                     targetOutput: BDV[Double],
                                     outputs: Array[BDV[Double]]):
  (Array[BDM[Double]], Array[BDV[Double]]) = {
    /* error back propagation */
    val errors = new Array[BDV[Double]](layers.size)
    for(i <- (layers.size - 1) until (0, -1)){
      val onesVector = BDV.ones[Double](outputs(i).length)
      val oSigmoid = Bsigmoid(outputs(i))
      //val outPrime = (onesVector :- outputs(i)) :* outputs(i)
      val outPrime = (onesVector :- oSigmoid) :* oSigmoid
      if(i == layers.size - 1){
        errors(i) = (outputs(i) :- targetOutput) :* outPrime
      }else{
        errors(i) = (weightMatrices(i + 1).t * errors(i + 1)) :* outPrime
      }
    }
    /* gradient */
    val gradientMatrices = new Array[BDM[Double]](layers.size)
    for(i <- (layers.size - 1) until (0, -1)) {
      gradientMatrices(i) = errors(i) * outputs(i - 1).t
    }
    (gradientMatrices, errors)
  }
}

/**
 * ::Experimental::
 * Trait for label conversion
 * for neural network classifier
 */
@Experimental
private[classification] trait LabelConverter {

  val labels: Set[Double]
  private val label2Index = labels.zipWithIndex.toMap
  private val index2Label = label2Index.map(_.swap)
  private val resultCount = label2Index.size

  /**
   * Returns a vector of double that has 1.0
   * at the index equals to label.toInt and
   * other element are zero
   * When resultCount is 2 then
   * returns a vector of size 1 with label
   * @param label label
   */
  protected def label2Vector(label: Double): BDV[Double] = {
    if(resultCount == 2){
      val result = BDV.zeros[Double](1)
      result(0) = label2Index(label).toDouble
      result
    }else{
      val result = BDV.zeros[Double](resultCount)
      result(label2Index(label)) = 1.0
      result
    }
  }

  /**
   * Returns a label that corresponds to the
   * array of double
   * @param resultVector vector that represents a label for neural network
   */
  protected def vector2Label(resultVector: BDV[Double]): Double = {
    require(resultVector.length == resultCount ||
      (resultVector.length == 1 && resultCount == 2))
    if(resultCount == 2) index2Label(round(resultVector(0)).toInt) else
      index2Label(Bargmax(resultVector))
  }
}

private class NeuralNetworkClassifierGradient(val layers: Array[Int], val labels: Set[Double])
  extends Gradient with LabelConverter with NeuralHelper {

  override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector):
    (linalg.Vector, Double) = {
    /* NB! weightMarices, gradientMatrices, errors have NULL zero element for addressing convenience */
    val (weightMatrices, bias) = unrollWeights(weights)
    /* forward run */
    val outputs = forwardRun(data.toBreeze.toDenseVector, weightMatrices, bias)
    /* error back propagation */
    val targetVector = label2Vector(label)
    val (gradientMatrices, errors) = wGradient(weightMatrices, targetVector, outputs)
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

private class NeuralNetworkGradient(val layers: Array[Int], val targetIndex: Int)
  extends Gradient with NeuralHelper {

  override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector):
  (linalg.Vector, Double) = {
    /* NB! weightMarices, gradientMatrices, errors have NULL zero element for addressing convenience */
    val (weightMatrices, bias) = unrollWeights(weights)
    /* forward run */
    val dataArray = data.toArray
    /* input == 0 for Autoencoder case */
    val input = if(targetIndex == 0) data.toBreeze.toDenseVector else
      new BDV[Double](dataArray, 0, 1, targetIndex)
    val targetVector = if(targetIndex == 0) input else
      new BDV[Double](dataArray, targetIndex)
    val outputs = forwardRun(input, weightMatrices, bias)
    /* error back propagation */
    val (gradientMatrices, errors) = wGradient(weightMatrices, targetVector, outputs)
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
//    println("Step:" + iter + " norm:" + Bnorm(gradient.toBreeze.toDenseVector))
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-stepSize, gradient.toBreeze, brzWeights)
    (Vectors.fromBreeze(brzWeights), 0)
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
class NeuralNetworkClassifierModel(val layers: Array[Int], val weights: linalg.Vector, val labels: Set[Double])
  extends ClassificationModel with LabelConverter with NeuralHelper with Serializable {

  private val (weightArray, bias) = unrollWeights(weights)

  override def predict(testData: RDD[linalg.Vector]): RDD[Double] = {
    testData.map(predict(_))
  }

  override def predict(testData: linalg.Vector): Double = {
    val outArray = forwardRun(testData.toBreeze.toDenseVector, weightArray, bias)
    vector2Label(outArray(layers.size - 1))
  }
}

class NeuralNetworkModel(val layers: Array[Int], val weights: linalg.Vector)
  extends NeuralHelper with Serializable {
  private val (weightArray, bias) = unrollWeights(weights)
  def propagate(data: linalg.Vector): Array[BDV[Double]] = forwardRun(data.toBreeze.toDenseVector, weightArray, bias)
}


/**
 * ::Experimental::
 * Trains Neural network classifier
 * NOTE: labels should represent the class index
 */
@Experimental
class NeuralNetworkClassifier private (hiddenLayers: Array[Int], numIterations: Int, learningRate: Double)
  extends Serializable {

  def run(data: RDD[LabeledPoint]) = {
    val labels = data.map( lp => lp.label).distinct().collect().toSet
    /* TODO: use LabelConverter instead! */
    val outputCount = if(labels.size == 2) 1 else labels.size
    val featureCount: Int = data.first().features.size
    val hl = if (hiddenLayers.size == 0) Array((featureCount + outputCount) / 2) else hiddenLayers
    val layers: Array[Int] = featureCount +: hl :+ outputCount
    val gradient = new NeuralNetworkClassifierGradient(layers, labels)
    val updater = new GradientUpdater()
    val gradientDescent = new GradientDescent(gradient, updater)
    gradientDescent.setNumIterations(numIterations).setStepSize(learningRate)
    val tupleData = data.map(lp => (lp.label, lp.features))
    val weights = gradientDescent.optimize(tupleData, gradient.initialWeights)
    new NeuralNetworkClassifierModel(layers, weights, labels)
  }
}

/**
 * ::Experimental::
 * Fabric for Neural Network classifier
 */
@Experimental
object NeuralNetworkClassifier {

  /**
   * Trains a Neural Network model given an RDD of `(label, features)` pairs.
   *
   * @param data RDD of `(label, array of features)` pairs.
   * @param hiddenLayers array of hidden layers sizes
   * @param numIterations number of iterations
   * @param learningRate learning rate
   */
  def train(data: RDD[LabeledPoint], hiddenLayers: Array[Int],
            numIterations: Int, learningRate: Double) : NeuralNetworkClassifierModel = {
    val nn = new NeuralNetworkClassifier(hiddenLayers, numIterations, learningRate)
    nn.run(data)
  }

  /**
   * Trains a Neural Network model given an RDD of `(label, features)` pairs.
   * Does 1000 iterations with learning rate 0.3
   *
   * @param data RDD of `(label, array of features)` pairs.
   * @param hiddenLayers array of hidden layers sizes
   */
  def train(data: RDD[LabeledPoint], hiddenLayers: Array[Int]) : NeuralNetworkClassifierModel = {
    train(data, hiddenLayers, 1000, 0.3)
  }

  /**
   * Trains a Neural Network model given an RDD of `(label, features)` pairs.
   * Creates one hidden layer of size = (input + ouput) / 2
   * Does 1000 iterations with learning rate 0.3
   *
   * @param data RDD of `(label, array of features)` pairs.
   */
  def train(data: RDD[LabeledPoint]) : NeuralNetworkClassifierModel = {
    train(data, Array(0), 1000, 0.3)
  }
}

/**
 * ::Experimental::
 * Trains neural network
 */
@Experimental
class NeuralNetwork private (hiddenLayers: Array[Int], numIterations: Int, learningRate: Double)
  extends Serializable {

  def run(data: RDD[(linalg.Vector, linalg.Vector)]) = {
    val outputCount: Int = data.first()._2.size
    val featureCount: Int = data.first()._1.size
    val hl = if (hiddenLayers.size == 0) Array((featureCount + outputCount) / 2) else hiddenLayers
    val layers: Array[Int] = featureCount +: hl :+ outputCount
    val gradient = new NeuralNetworkGradient(layers, featureCount)
    val updater = new GradientUpdater()
    val gradientDescent = new GradientDescent(gradient, updater)
    gradientDescent.setNumIterations(numIterations).setStepSize(learningRate)
    val tupleData = data.map{ case(in, out) =>
      (0.0, Vectors.fromBreeze(BDV.vertcat(in.toBreeze.toDenseVector, out.toBreeze.toDenseVector)))}
    val weights = gradientDescent.optimize(tupleData, gradient.initialWeights)
    new NeuralNetworkModel(layers, weights)
  }
}

/**
 * ::Experimental::
 * Fabric for neural network
 */
@Experimental
object NeuralNetwork {

  /**
   * Trains a neural network model given an RDD of `(input, output)` pairs.
   *
   * @param data RDD of `(input, output)` pairs of vectors.
   * @param hiddenLayers array of hidden layers sizes
   * @param numIterations number of iterations
   * @param learningRate learning rate
   */
  def train(data: RDD[(linalg.Vector, linalg.Vector)], hiddenLayers: Array[Int],
            numIterations: Int, learningRate: Double) : NeuralNetworkModel = {
    val nn = new NeuralNetwork(hiddenLayers, numIterations, learningRate)
    nn.run(data)
  }

  /**
   * Trains a Neural Network model given an RDD of `(input, output)` pairs.
   * Does 1000 iterations with learning rate 0.3
   *
   * @param data RDD of `(input, output)` pairs.
   * @param hiddenLayers array of hidden layers sizes
   */
  def train(data: RDD[(linalg.Vector, linalg.Vector)], hiddenLayers: Array[Int]) : NeuralNetworkModel = {
    train(data, hiddenLayers, 1000, 0.3)
  }

  /**
   * Trains a neural network model given an RDD of `(input, output)` pairs.
   * Creates one hidden layer of size = (input + ouput) / 2
   * Does 1000 iterations with learning rate 0.3
   *
   * @param data RDD of `(input, output)` pairs.
   */
  def train(data: RDD[(linalg.Vector, linalg.Vector)]) : NeuralNetworkModel = {
    train(data, Array(0), 1000, 0.3)
  }
}
