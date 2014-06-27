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

package org.apache.spark.mllib.regression

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.{Logging, SparkException}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.util.MLUtils._

/**
 * :: DeveloperApi ::
 * GeneralizedLinearModel (GLM) represents a model trained using
 * GeneralizedLinearAlgorithm. GLMs consist of a weight vector and
 * an intercept.
 *
 * @param weights Weights computed for every feature.
 * @param intercept Intercept computed for this model.
 */
@DeveloperApi
abstract class GeneralizedLinearModel(val weights: Vector, val intercept: Double)
  extends Serializable {

  /** Whether to add intercept (default: false). */
  var addIntercept: Boolean = false

  /**
   * Predict the result given a data point and the weights learned.
   *
   * @param dataMatrix Row vector containing the features for this data point
   * @param weightMatrix Column vector containing the weights of the model
   * @param intercept Intercept of the model.
   */
  protected def predictPoint(dataMatrix: Vector, weightMatrix: Vector, intercept: Double): Double

  /**
   * Predict values for the given data set using the model trained.
   *
   * @param testData RDD representing data points to be predicted
   * @return RDD[Double] where each entry contains the corresponding prediction
   */
  def predict(testData: RDD[Vector]): RDD[Double] = {
    // A small optimization to avoid serializing the entire model. Only the weightsMatrix
    // and intercept is needed.
    val localWeights = weights
    val localIntercept = intercept

    testData.map(v => predictPoint(v, localWeights, localIntercept))
  }

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param testData array representing a single data point
   * @return Double prediction from the trained model
   */
  def predict(testData: Vector): Double = {
    predictPoint(testData, weights, intercept)
  }
}

/**
 * :: DeveloperApi ::
 * GeneralizedLinearAlgorithm implements methods to train a Generalized Linear Model (GLM).
 * This class should be extended with an Optimizer to create a new GLM.
 */
@DeveloperApi
abstract class GeneralizedLinearAlgorithm[M <: GeneralizedLinearModel]
  extends Logging with Serializable {

  protected def validators: Seq[RDD[LabeledPoint] => Boolean] = List()

  /** The optimizer to solve the problem. */
  def optimizer: Optimizer

  /** Whether to add intercept (default: false). */
  protected var addIntercept: Boolean = false

  /**
   * This model contains multiple hyperplanes, which means multiple intercepts are required.
   * Since it can not be stored in a single intercept variable, let's keep it inside weights vector
   * and set the intercept variable to zero. (default: false).
   */
  protected def isMultipleIntercepts: Boolean = numOfIntercepts > 1

  /**
   * The number of intercepts in this model, (default: 1).
   */
  protected var numOfIntercepts: Int = 1

  protected var validateData: Boolean = true

  /**
   * Create a model given the weights and intercept
   */
  protected def createModel(weights: Vector, intercept: Double): M

  /**
   * Set if the algorithm should add an intercept. Default false.
   * We set the default to false because adding the intercept will cause memory allocation.
   */
  def setIntercept(addIntercept: Boolean): this.type = {
    this.addIntercept = addIntercept
    this
  }

  /**
   * Set if the algorithm should validate data before training. Default true.
   */
  def setValidateData(validateData: Boolean): this.type = {
    this.validateData = validateData
    this
  }

  /**
   * Run the algorithm with the configured parameters on an input
   * RDD of LabeledPoint entries.
   */
  def run(input: RDD[LabeledPoint]): M = {
    val numFeatures: Int = input.first().features.size
    val dimOfWeights = if(isMultipleIntercepts) (numFeatures + 1) * numOfIntercepts else numFeatures
    val initialWeights = Vectors.dense(new Array[Double](dimOfWeights))
    run(input, initialWeights)
  }

  /**
   * Run the algorithm with the configured parameters on an input RDD
   * of LabeledPoint entries starting from the initial weights provided.
   */
  def run(input: RDD[LabeledPoint], initialWeights: Vector): M = {

    // Check the data properties before running the optimizer
    if (validateData && !validators.forall(func => func(input))) {
      throw new SparkException("Input validation failed.")
    }

    // Prepend an extra variable consisting of all 1.0's for the intercept.
    val data = if (addIntercept) {
      input.map(labeledPoint => (labeledPoint.label, appendBias(labeledPoint.features)))
    } else {
      input.map(labeledPoint => (labeledPoint.label, labeledPoint.features))
    }

    // TODO: We need a nicer api to allow users to set the initial intercept.
    // Better by unifying the intercept and weights entirely into one vector make it cleaner.
    // PS, if isMultipleIntercepts == true, users can set the intercepts by changing initialWeights
    // which is different behavior when only having single intercept. We need to address this.
    val initialWeightsWithIntercept = if (addIntercept && !isMultipleIntercepts) {
      appendBias(initialWeights)
    } else {
      initialWeights
    }

    assert(initialWeightsWithIntercept.toBreeze.length % data.first._2.toBreeze.length == 0)

    val weightsWithIntercept = optimizer.optimize(data, initialWeightsWithIntercept)

    // If the dimension of weightsWithIntercept / dimOfData != 1, the model have multiple
    // hyperplane, which means the multiple intercepts can not be stored in single intercept
    // variable. We just keep it in weights vector and set the intercept variable to zero.
//    val isMultipleIntercepts =
//      if(weightsWithIntercept.toBreeze.length / dimOfData == 1) false else true

    val intercept =
      if (addIntercept && !isMultipleIntercepts) {
        weightsWithIntercept(weightsWithIntercept.size - 1)
      }
      else {
        0.0
      }

    val weights =
      if (addIntercept && !isMultipleIntercepts) {
        Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1))
      } else {
        weightsWithIntercept
      }

    val model = createModel(weights, intercept)
    model.addIntercept = this.addIntercept
    model
  }
}
