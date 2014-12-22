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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.rdd.RDD

/**
 * Classification model trained using Multinomial Logistic Regression.
 *
 * @param weights Weights computed for every feature.
 * @param intercept Intercept computed for this model.
 */
class LogisticRegressionModel (
    override val weights: Vector,
    override val intercept: Double)
  extends GeneralizedLinearModel(weights, intercept) with ClassificationModel with Serializable {

  private var threshold: Option[Double] = Some(0.5)

  /**
   * :: Experimental ::
   * Sets the threshold that separates positive predictions from negative predictions. An example
   * with prediction score greater than or equal to this threshold is identified as an positive,
   * and negative otherwise. The default value is 0.5.
   */
  @Experimental
  def setThreshold(threshold: Double): this.type = {
    this.threshold = Some(threshold)
    this
  }

  /**
   * :: Experimental ::
   * Clears the threshold so that `predict` will output raw prediction scores.
   */
  @Experimental
  def clearThreshold(): this.type = {
    threshold = None
    this
  }

  override protected def predictPoint(dataMatrix: Vector, weightMatrix: Vector,
      intercept: Double) = {

    val brzDataMatrix = dataMatrix.toBreeze
    val brzWeightMatrix = weightMatrix.toBreeze

    // If dataMatrix and weightMatrix have the same dimension, it's binary
    // logistic regression.
    if(brzDataMatrix.length == brzWeightMatrix.length){
      val margin = brzDataMatrix.dot(brzWeightMatrix) + intercept
      val score = 1.0 / (1.0 + math.exp(-margin))
      threshold match {
        case Some(t) => if (score < t) 0.0 else 1.0
        case None => score
      }
    } else {
      // Prefer to use appendBias instead of prependOne for keeping the index the same.
      val brzDataWithIntercept =
        if(addIntercept == true) appendBias(dataMatrix).toBreeze else brzDataMatrix

      assert((brzWeightMatrix.length % brzDataWithIntercept.length) == 0)

      val nClasses = (brzWeightMatrix.length / brzDataWithIntercept.length) + 1
      val probs = new Array[Double](nClasses)

      probs(0) = 1

      var denominator = 1.0
      for (i <- 0 until (nClasses - 1)) {
        var acc = 0.0
        brzDataWithIntercept.activeIterator.foreach {
          case (_, 0.0) => // Skip explicit zero elements.
          case (j, value) => acc += value * brzWeightMatrix((i * brzDataWithIntercept.length) + j)
        }
        probs(i + 1) = math.exp(acc)
        denominator += probs(i + 1)
      }

      for (i <- 0 until nClasses) {
        probs(i) /= denominator
      }

      // Should have another predictPointProb api
      var prediction = 0
      var max = probs(0)
      for (i <- 1 until probs.length) {
        if (probs(i) > max) {
          prediction = i
          max = probs(i)
        }
      }
      prediction.toDouble

    }
  }
}

/**
<<<<<<< HEAD
 * Train a classification model for Logistic Regression using Stochastic Gradient Descent.
 * NOTE: Labels used in Logistic Regression should be {0, 1}
 *
 * Using [[LogisticRegressionWithLBFGS]] is recommended over this.
=======
 * Train a classification model for Multinomial Logistic Regression
 * using Stochastic Gradient Descent.
 * NOTE: Labels used in Multinomial Logistic Regression should be {0, 1, ..., k - 1}
 * for k classes classification problem.
 * By default, the number of classes are 2 (binary classification).
>>>>>>> dbtsai-mlor
 */
class LogisticRegressionWithSGD private (
    private var stepSize: Double,
    private var numIterations: Int,
    private var regParam: Double,
    private var miniBatchFraction: Double,
    private var classes: Int = 2)
  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

  private val gradient = new LogisticGradient()
  private val updater = new SquaredL2Updater()
  override val optimizer = new LBFGS(gradient, updater)
    //.setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setRegParam(regParam)
    //.setMiniBatchFraction(miniBatchFraction)

  this.numOfIntercepts = classes - 1

  // Since numOfIntercepts can be changed by `setNumOfClasses`, we make it as a `def`.
  override protected def validators =
    List(DataValidators.multinomialLabelValidator(this.numOfIntercepts + 1))

  /**
   * Construct a LogisticRegression object with default parameters
   */
  def this() = this(1.0, 100, 0.0, 1.0)

  /**
   * Set the number of classes for Multinomial Logistic Regression.
   * By default, it's 2
   */
  def setNumOfClasses(classes: Int): this.type = {
    this.numOfIntercepts = classes - 1
    this
  }

  override protected def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}

/**
<<<<<<< HEAD
 * Top-level methods for calling Logistic Regression using Stochastic Gradient Descent.
 * NOTE: Labels used in Logistic Regression should be {0, 1}
=======
 * Top-level methods for calling Multinomial Logistic Regression.
 * NOTE: Labels used in Multinomial Logistic Regression should be {0, 1, ..., k - 1}
 * for k classes classification problem.
>>>>>>> dbtsai-mlor
 */
object LogisticRegressionWithSGD {
  // NOTE(shivaram): We use multiple train methods instead of default arguments to support
  // Java programs.

  /**
   * Train a binary logistic regression model given an RDD of (label, features) pairs.
   * We run a fixed
   * number of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
   * gradient descent are initialized using the initial weights provided.
   * NOTE: Labels used in Logistic Regression should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   * @param initialWeights Initial set of weights to be used. Array should be equal in size to
   *        the number of features in the data.
   */
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      miniBatchFraction: Double,
      initialWeights: Vector): LogisticRegressionModel = {
    new LogisticRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction)
      .run(input, initialWeights)
  }

  /**
   * Train a multinomial logistic regression model given an RDD of (label, features) pairs.
   * We run a fixed
   * number of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
   * gradient descent are initialized using the initial weights provided.
   * NOTE: Labels used in Logistic Regression should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   * @param initialWeights Initial set of weights to be used. Array should be equal in size to
   *        the number of features in the data.
   * @param classes the number of classes in multi-classification problem
   */
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      miniBatchFraction: Double,
      initialWeights: Vector,
      classes: Int): LogisticRegressionModel = {
    new LogisticRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction, classes)
      .run(input, initialWeights)
  }

  /**
   * Train a binary logistic regression model given an RDD of (label, features) pairs.
   * We run a fixed
   * number of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient.
   * NOTE: Labels used in Logistic Regression should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   */
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      miniBatchFraction: Double): LogisticRegressionModel = {
    new LogisticRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction)
      .run(input)
  }

  /**
   * Train a multinomial logistic regression model given an RDD of (label, features) pairs.
   * We run a fixed
   * number of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient.
   * NOTE: Labels used in Logistic Regression should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   * @param classes the number of classes in multi-classification problem
   */
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      miniBatchFraction: Double,
      classes: Int): LogisticRegressionModel = {
    new LogisticRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction, classes)
      .run(input)
  }

  /**
   * Train a binary logistic regression model given an RDD of (label, features) pairs.
   * We run a fixed
   * number of iterations of gradient descent using the specified step size. We use the entire data
   * set to update the gradient in each iteration.
   * NOTE: Labels used in Logistic Regression should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of Gradient Descent.
   * @return a LogisticRegressionModel which has the weights and offset from training.
   */
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double): LogisticRegressionModel = {
    train(input, numIterations, stepSize, 1.0)
  }

   /**
   * Train a multinomial logistic regression model given an RDD of (label, features) pairs.
    * We run a fixed
   * number of iterations of gradient descent using the specified step size.
    * We use the entire data
   * set to update the gradient in each iteration.
   * NOTE: Labels used in Logistic Regression should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of Gradient Descent.
   * @param classes the number of classes in multi-classification problem
   * @return a LogisticRegressionModel which has the weights and offset from training.
   */
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      classes: Int): LogisticRegressionModel = {
    train(input, numIterations, stepSize, 1.0, classes)
  }


  /**
   * Train a binary logistic regression model given an RDD of (label, features) pairs.
   * We run a fixed
   * number of iterations of gradient descent using a step size of 1.0. We use the entire data set
   * to update the gradient in each iteration.
   * NOTE: Labels used in Logistic Regression should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a LogisticRegressionModel which has the weights and offset from training.
   */
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int): LogisticRegressionModel = {
    train(input, numIterations, 1.0, 1.0)
  }

  /**
   * Train a multinomial logistic regression model given an RDD of (label, features) pairs.
   * We run a fixed
   * number of iterations of gradient descent using a step size of 1.0. We use the entire data set
   * to update the gradient in each iteration.
   * NOTE: Labels used in Logistic Regression should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param classes the number of classes in multi-classification problem
   * @return a LogisticRegressionModel which has the weights and offset from training.
   */
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      classes: Int): LogisticRegressionModel = {
    train(input, numIterations, 1.0, 1.0, classes)
  }

}

/**
 * Train a classification model for Logistic Regression using Limited-memory BFGS.
 * Standard feature scaling and L2 regularization are used by default.
 * NOTE: Labels used in Logistic Regression should be {0, 1}
 */
class LogisticRegressionWithLBFGS
  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

  this.setFeatureScaling(true)

  override val optimizer = new LBFGS(new LogisticGradient, new SquaredL2Updater)

  override protected val validators = List(DataValidators.binaryLabelValidator)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}
