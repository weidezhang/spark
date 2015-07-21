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

package org.apache.spark.ml.regression

import breeze.linalg.{argmax => Bargmax}

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.ann.{FeedForwardTopology, FeedForwardTrainer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame

/**
 * Params for Multilayer Perceptron.
 */
// TODO: ADD layers: Array[Int] and blockSize: Int params
private[ml] trait MultilayerPerceptronParams extends PredictorParams
with HasSeed with HasMaxIter with HasTol {
  /**
   * Layer sizes including input and output.
   * @group param
   */
  final val layers: IntArrayParam =
    // TODO: we need IntegerArrayParam!
    new IntArrayParam(this, "layers",
      "Sizes of layers including input and output from bottom to the top." +
      " E.g., Array(780, 100, 10) means 780 inputs, " +
      "hidden layer with 100 neurons and output layer of 10 neurons."
      // TODO: how to check that array is not empty?
      )

  /**
   * Block size for stacking input data in matrices. Speeds up the computations
   * @group expertParam
   */
  final val blockSize: IntParam = new IntParam(this, "blockSize",
    "Block size for stacking input data in matrices.",
    ParamValidators.gt(0))

  /** @group setParam */
  def setLayers(value: Array[Int]): this.type = set(layers, value)

  /** @group getParam */
  final def getLayers: Array[Int] = $(layers)

  /** @group setParam */
  def setBlockSize(value: Int): this.type = set(blockSize, value)

  /** @group getParam */
  final def getBlockSize: Int = $(blockSize)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-4.
   * @group setParam
   */
  def setTol(value: Double): this.type = set(tol, value)

  /**
   * Set the seed for weights initialization.
   * Default is 11L.
   * @group setParam
   */
  def setSeed(value: Long): this.type = set(seed, value)

  setDefault(seed -> 11L, maxIter -> 100, tol -> 1e-4, layers -> Array(1, 1), blockSize -> 1)
}

/**
 * :: Experimental ::
 * Multi-layer perceptron regression. Contains sigmoid activation function on all layers.
 * See https://en.wikipedia.org/wiki/Multilayer_perceptron for details.
 *
 */
@Experimental
class MultilayerPerceptronRegressor (override val uid: String)
  extends Regressor[Vector, MultilayerPerceptronRegressor, MultilayerPerceptronRegressorModel]
  with MultilayerPerceptronParams with Logging {

  def this() = this(Identifiable.randomUID("mlpr"))

  override def copy(extra: ParamMap): MultilayerPerceptronRegressor = defaultCopy(extra)

  /**
   * Train a model using the given dataset and parameters.
   * Developers can implement this instead of [[fit()]] to avoid dealing with schema validation
   * and copying parameters into the model.
   *
   * @param dataset  Training dataset
   * @return  Fitted model
   */
  override protected def train(dataset: DataFrame): MultilayerPerceptronRegressorModel = {
    // TODO: find a better way to get Vectors
    val data = dataset.map { x => (x.getAs[Vector](0), x.getAs[Vector](1)) }
    val myLayers = getLayers
    val topology = FeedForwardTopology.multiLayerPerceptron(myLayers, false)
    val FeedForwardTrainer = new FeedForwardTrainer(topology, myLayers(0), myLayers.last)
    FeedForwardTrainer.LBFGSOptimizer.setConvergenceTol(getTol).setNumIterations(getMaxIter)
    FeedForwardTrainer.setBatchSize(getBlockSize)
    val mlpModel = FeedForwardTrainer.train(data)
    new MultilayerPerceptronRegressorModel(uid, myLayers, mlpModel.weights())
  }
}

/**
 * :: Experimental ::
 * Multi-layer perceptron regression model.
 *
 * @param layers array of layer sizes including input and output
 * @param weights weights or parameters of the model
 */
@Experimental
class MultilayerPerceptronRegressorModel private[ml] (override val uid: String,
                                                      layers: Array[Int],
                                                      weights: Vector)
  extends RegressionModel[Vector, MultilayerPerceptronRegressorModel]{

  private val mlpModel =
    FeedForwardTopology.multiLayerPerceptron(layers, false).getInstance(weights)

  /**
   * Predict label for the given features.
   * This internal method is used to implement [[transform()]] and output [[predictionCol]].
   */
  override protected def predict(features: Vector): Double = {
    Bargmax(predictVector(features).toArray).toDouble
  }

  /**
   * Predict output for the given input.
   * @param feautres features vector
   */
  def predictVector(feautres: Vector): Vector = {
    mlpModel.predict(feautres)
  }

  override def copy(extra: ParamMap): MultilayerPerceptronRegressorModel = {
    copyValues(new MultilayerPerceptronRegressorModel(uid, layers, weights), extra)
  }
}
