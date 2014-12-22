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

package org.apache.spark.mllib.optimization


import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseVector}
import org.apache.spark.mllib.linalg.BLAS.{axpy, dot, scal}

/**
 * :: DeveloperApi ::
 * Class used to compute the gradient for a loss function, given a single data point.
 */
@DeveloperApi
abstract class Gradient extends Serializable {
  /**
   * Compute the gradient and loss given the features of a single data point.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   *
   * @return (gradient: Vector, loss: Double)
   */
  def compute(data: Vector, label: Double, weights: Vector): (Vector, Double)

  /**
   * Compute the gradient and loss given the features of a single data point,
   * add the gradient to a provided vector to avoid creating new objects, and return loss.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   * @param cumGradient the computed gradient will be added to this vector
   *
   * @return loss
   */
  def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double
}

/**
 * :: DeveloperApi ::
 * Compute gradient and loss for a multinomial logistic loss function,
 * as used in multi-class classification (it is also used in binary logistic regression).
 * See also the documentation for the precise formulation.
 */
@DeveloperApi
class LogisticGradient extends Gradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  override def compute(
                        data: Vector,
                        label: Double,
                        weights: Vector,
                        cumGradient: Vector): Double = {
    assert((weights.size % data.size) == 0)
    val dataSize = data.size
    // (n + 1) is number of classes
    val n = (weights.size / dataSize)
    val numerators = Array.ofDim[Double](n)

    var denominator = 0.0
    var margin = 0.0

    val weightsArray = weights match {
      case dv: DenseVector => dv.values
      case _ =>
        throw new IllegalArgumentException(
          s"weights only supports dense vector but got type ${weights.getClass}.")
    }
    val cumGradientArray = cumGradient match {
      case dv: DenseVector => dv.values
      case _ =>
        throw new IllegalArgumentException(
          s"cumGradient only supports dense vector but got type ${cumGradient.getClass}.")
    }

    var i = 0
    while (i < n) {
      var sum = 0.0
      data.foreachActive { (index, value) =>
        if (value != 0.0) sum += value * weightsArray((i * dataSize) + index)
      }
      if (i == label.toInt - 1) margin = sum
      numerators(i) = math.exp(sum)
      denominator += numerators(i)
      i += 1
    }

    i = 0
    while (i < n) {
      val multiplier = numerators(i) / (denominator + 1.0) - {
        if (label != 0.0 && label == i + 1) 1.0 else 0.0
      }
      data.foreachActive { (index, value) =>
        if (value != 0.0) cumGradientArray(i * dataSize + index) += multiplier * value
      }
      i += 1
    }

    if (label > 0.0) {
      math.log1p(denominator) - margin
    } else {
      math.log1p(denominator)
    }
  }
}
/*@DeveloperApi
class LogisticGradient extends Gradient {

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val margin = -1.0 * dot(data, weights)
    val gradientMultiplier = (1.0 / (1.0 + math.exp(margin))) - label
    val gradient = data.copy
    scal(gradientMultiplier, gradient)
    val loss =
      if (label > 0) {
        math.log1p(math.exp(margin)) // log1p is log(1+p) but more accurate for small p
      } else {
        math.log1p(math.exp(margin)) - margin
      }

    (gradient, loss)
  }


  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {

    def alpha(i: Int): Int = if (i == 0) 1 else 0
    def delta(i: Int, j: Int): Int = if (i == j) 1 else 0

    val brzData = data.toBreeze
    val brzWeights = weights.toBreeze
    val brzCumGradient = cumGradient.toBreeze

    assert((brzWeights.length % brzData.length) == 0)
    assert(cumGradient.toBreeze.length == brzWeights.length)

    val nClasses = (brzWeights.length / brzData.length) + 1
    val classLabel = math.round(label).toInt

    var denominator = 1.0
    val numerators = Array.ofDim[Double](nClasses - 1)

    var i = 0
    while (i < nClasses - 1) {
      var acc = 0.0
      brzData.activeIterator.foreach {
        case (_, 0.0) => // Skip explicit zero elements.
        case (j, value) => acc += value * brzWeights((i * brzData.length) + j)
      }
      numerators(i) = math.exp(acc)
      denominator += numerators(i)
      i += 1
    }

    i = 0
    while (i < nClasses - 1) {
      brzData.activeIterator.foreach {
        case (_, 0.0) => // Skip explicit zero elements.
        case (j, value) => brzCumGradient(i * data.toBreeze.length + j) -=
          ((1 - alpha(classLabel)) * delta(classLabel, i + 1) - numerators(i) / denominator) *
            brzData(j)
      }
      i += 1
    }

    classLabel match {
      case 0 => -math.log(1.0 / denominator)
      case _ => -math.log(numerators(classLabel - 1) / denominator)

    }
  }
}
*/
/**
 * :: DeveloperApi ::
 * Compute gradient and loss for a Least-squared loss function, as used in linear regression.
 * This is correct for the averaged least squares loss function (mean squared error)
 *              L = 1/n ||A weights-y||^2
 * See also the documentation for the precise formulation.
 */
@DeveloperApi
class LeastSquaresGradient extends Gradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val diff = dot(data, weights) - label
    val loss = diff * diff
    val gradient = data.copy
    scal(2.0 * diff, gradient)
    (gradient, loss)
  }

  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    val diff = dot(data, weights) - label
    axpy(2.0 * diff, data, cumGradient)
    diff * diff
  }
}

/**
 * :: DeveloperApi ::
 * Compute gradient and loss for a Hinge loss function, as used in SVM binary classification.
 * See also the documentation for the precise formulation.
 * NOTE: This assumes that the labels are {0,1}
 */
@DeveloperApi
class HingeGradient extends Gradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val dotProduct = dot(data, weights)
    // Our loss function with {0, 1} labels is max(0, 1 - (2y - 1) (f_w(x)))
    // Therefore the gradient is -(2y - 1)*x
    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      val gradient = data.copy
      scal(-labelScaled, gradient)
      (gradient, 1.0 - labelScaled * dotProduct)
    } else {
      (Vectors.sparse(weights.size, Array.empty, Array.empty), 0.0)
    }
  }

  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    val dotProduct = dot(data, weights)
    // Our loss function with {0, 1} labels is max(0, 1 - (2y - 1) (f_w(x)))
    // Therefore the gradient is -(2y - 1)*x
    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      axpy(-labelScaled, data, cumGradient)
      1.0 - labelScaled * dotProduct
    } else {
      0.0
    }
  }
}
