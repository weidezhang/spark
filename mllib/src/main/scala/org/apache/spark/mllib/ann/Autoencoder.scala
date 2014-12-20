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

package org.apache.spark.mllib.ann

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification.{ANNClassifierModel, ANNClassifier}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
 * :: Experimental ::
 * Implements autoencoder
 * http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders
 */
@Experimental
object Autoencoder {

  /**
   * Pre-trains weights for the hidden layer by means of autoencoder,
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param data RDD containing vectors for training.
   * @param hiddenLayer size of hidden layer
   * @param maxIterations number of iterations
   * @return ANN model.
   */
  def train(data: RDD[Vector], hiddenLayer: Int,
          maxIterations: Int): ArtificialNeuralNetworkModel = {
    /* TODO: remove data duplication */
    val training = data.map( x => (x, x))
    ArtificialNeuralNetwork.train(training, Array(hiddenLayer), maxIterations)
  }

  /**
   * Trains pre-trained autoencoder model,
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param data RDD containing vectors for training.
   * @param model pre-trained ANN model
   * @param maxIterations number of iterations
   * @return ANN model.
   */
  def train(data: RDD[Vector], model: ArtificialNeuralNetworkModel,
            maxIterations: Int): ArtificialNeuralNetworkModel = {
    /* TODO: remove data duplication */
    val training = data.map( x => (x, x))
    ArtificialNeuralNetwork.train(training, model, maxIterations)
  }
}

class StackedAutoencoderModel(val intermediateModels: Array[ArtificialNeuralNetworkModel],
                               val topModel: ANNClassifierModel) extends Serializable {
  lazy val classifierModel: ANNClassifierModel = {
    /* TODO: require models consistency */
    val hiddenLayersTopology = new ArrayBuffer[Int]()
    for(i <- 0 until intermediateModels.length){
      hiddenLayersTopology += intermediateModels(i).topology(1)
    }
    val topology =
      intermediateModels(0).topology(0) +: hiddenLayersTopology :+ topModel.annModel.topology.last
    /* TODO: extract and use a method that computes the size of the weights vector */
    val finalWeights = ArtificialNeuralNetwork.randomWeights(topology(0), topology.last,
      hiddenLayersTopology.toArray, 11).toBreeze
    /* TODO: remove conversions! */
    var offset = 0
    for(i <- 0 until intermediateModels.length){
      val weights: Vector = intermediateModels(i).weightsByLayer(1)
      finalWeights(offset until (offset + weights.size)) := weights.toBreeze.toDenseVector
      offset += weights.size
    }
    val topWeights = topModel.annModel.weights
    finalWeights(offset until finalWeights.size) := topWeights.toBreeze.toDenseVector
    val annModel = new ArtificialNeuralNetworkModel(Vectors.dense(finalWeights.toArray),
      topology.toArray)
    new ANNClassifierModel(annModel, topModel.labelToIndex)
  }
}

/**
 * :: Experimental ::
 * Implements stacked autoencoder
 * http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders
 */
@Experimental
class StackedAutoencoder private(val hiddenLayersTopology: Array[Int],
                                 val maxIterations: Int) {

  /**
   * Pre-trains weights for all layers of ANN classifier
   * by means of stacked autoencoder.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param data RDD containing labeled points for training.
   * @return ANN classifier model.
   */
  def run(data: RDD[LabeledPoint]): StackedAutoencoderModel = {
    val intermediateModels =
      new Array[ArtificialNeuralNetworkModel](hiddenLayersTopology.size)
    var train = data.map(lp => lp.features)
    var test = data.map(lp => (lp.label, lp.features))
    /* loop for pre-training of the weight matrices of hidden layers */
    for(i <- 0 until hiddenLayersTopology.size){
      val model = Autoencoder.train(train, hiddenLayersTopology(i), maxIterations)
      intermediateModels(i) = model
      /* TODO: remove data duplication */
      test = test.map( x => (x._1, model.output(x._2, 1)))
      train = test.map( x => x._2)
    }
    /* train the last-layer classifier. has to be softmax regression */
    /* TODO: perform softmax training */
    val classTrain = test.map(x => new LabeledPoint(x._1, x._2))
    val topModel = ANNClassifier.train(classTrain, Array[Int](), maxIterations, 1.0, 1e-4)
    new StackedAutoencoderModel(intermediateModels, topModel)
  }

  /**
   * Trains the pre-trained stacked autoencoder
   * Uses default convergence tolerance 1e-4 for LBFGS.
   * This code is almost duplicate to the primary run code
   * The other option is to pass all weights to run
   * which looks really messy, so in this case
   * duplication is better. Hopefully, I'll figure out
   * how to reuse the code in both 'run's
   *
   * @param data RDD containing labeled points for training.
   * @param stackedAutoencoderModel stacked autoencode model
   * @return ANN classifier model.
   */
  def run(data: RDD[LabeledPoint],
          stackedAutoencoderModel: StackedAutoencoderModel): StackedAutoencoderModel = {
    val intermediateModels =
      new Array[ArtificialNeuralNetworkModel](hiddenLayersTopology.size)
    var train = data.map(lp => lp.features)
    var test = data.map(lp => (lp.label, lp.features))
    /* loop for pre-training of the weight matrices of hidden layers */
    for(i <- 0 until hiddenLayersTopology.size){
      val model =
        Autoencoder.train(train, stackedAutoencoderModel.intermediateModels(i), maxIterations)
      intermediateModels(i) = model
      /* TODO: remove data duplication */
      test = test.map( x => (x._1, model.output(x._2, 1)))
      train = test.map( x => x._2)
    }
    /* train the last-layer classifier. has to be softmax regression */
    /* TODO: perform softmax training */
    val classTrain = test.map(x => new LabeledPoint(x._1, x._2))
    val topModel =
      ANNClassifier.train(classTrain, stackedAutoencoderModel.topModel, maxIterations, 1.0, 1e-4)
    new StackedAutoencoderModel(intermediateModels, topModel)
  }
}

/**
 * :: Experimental ::
 * Top level methods for calling stacked autoencoder
 * http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders
 */
@Experimental
object StackedAutoencoder {

  /**
   * Pre-trains weights for all layers of ANN classifier
   * by means of stacked autoencoder.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param data RDD containing labeled points for training.
   * @param hiddenLayersTopology hidden layers topology
   * @param maxIterations number of iterations for each stack
   * @return ANN classifier model.
   */
  def train(data: RDD[LabeledPoint], hiddenLayersTopology: Array[Int],
            maxIterations: Int): StackedAutoencoderModel = {
    new StackedAutoencoder(hiddenLayersTopology, maxIterations).run(data)
  }

  def train(data: RDD[LabeledPoint], stackedModel: StackedAutoencoderModel,
            maxIterations: Int): StackedAutoencoderModel = {
    val classModel = stackedModel.classifierModel
    /* TODO: extract from ANNClassifier and use */
    val hiddenLayersTopology =
      classModel.annModel.topology.slice(1, classModel.annModel.topology.length - 1)
    new StackedAutoencoder(hiddenLayersTopology, maxIterations).run(data, stackedModel)
  }
}
