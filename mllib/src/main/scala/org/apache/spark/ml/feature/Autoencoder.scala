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

package org.apache.spark.ml.feature

import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.ann.{FeedForwardTrainer, FeedForwardTopology}
import org.apache.spark.ml.classification.MultilayerPerceptronParams
import org.apache.spark.ml.feature.InputDataType.InputDataType
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.param.{IntParam, Params, ParamMap}
import org.apache.spark.mllib.linalg.{VectorUDT, Vectors, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, DataFrame}
import org.apache.spark.sql.types.{StructField, StructType}

/**
 * Params for [[Autoencoder]] and [[AutoencoderModel]].
 */
private[feature] trait AutoencoderParams extends Params with HasInputCol with HasOutputCol {
}

/**
 * Input data types enum
 */
private[feature] object InputDataType extends Enumeration {
  type InputDataType = Value
  val Binary, Real01, Real = Value
}

/**
 * :: Experimental ::
 * Autoencoder.
 */
@Experimental
class Autoencoder (override val uid: String) extends Estimator[AutoencoderModel]
  with MultilayerPerceptronParams with AutoencoderParams  {

  def this() = this(Identifiable.randomUID("autoencoder"))

  // TODO: make sure that user understands how to set it
  /** @group setParam */
  def setLayers(value: Array[Int]): this.type = set(layers, value)

  /** @group setParam */
  def setBlockSize(value: Int): this.type = set(blockSize, value)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

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
   * @group setParam
   */
  def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Fits a model to the input data.
   */
  override def fit(dataset: DataFrame): AutoencoderModel = {
    val data = dataset.select($(inputCol)).map { case Row(x: Vector) => (x, x) }
    val myLayers = $(layers)
    // TODO: initialize topology based on the data type (binary, real [0..1], real)
    // binary => false + cross entropy
    // real [0..1] => false + sq error
    // real [0..1] that sum to one => true + cross entropy
    // real => remove the top layer + sq error
    // TODO: how to set one of the mentioned data types?
    val topology = FeedForwardTopology.multiLayerPerceptron(myLayers, false)
    val FeedForwardTrainer = new FeedForwardTrainer(topology, myLayers(0), myLayers.last)
    FeedForwardTrainer.LBFGSOptimizer.setConvergenceTol($(tol)).setNumIterations($(maxIter))
    FeedForwardTrainer.setStackSize($(blockSize))
    println(inputDataType(data).toString)
    val autoencoderModel = FeedForwardTrainer.train(data)
    // TODO: parameter for the layer which is supposed to be encoder output
    // in case of deep autoencoders
    // TODO: what about decoder back to normal?
    new AutoencoderModel(uid, myLayers.init, autoencoderModel.weights())
  }

  private def inputDataType(data: RDD[(Vector, Vector)]): InputDataType = {
    val (binary, real01) = data.map{ case(x, y) =>
      (x.toArray.forall(z => (z == 0.0 || z == 1.0)), x.toArray.forall(z => (z >= 0.0 && z <= 1.0)))
    }.reduce { case(p1, p2) =>
      (p1._1 && p2._1, p1._2 && p2._2)
    }
    if (binary) return InputDataType.Binary
    if (real01) return InputDataType.Real01
    InputDataType.Real
  }

  override def copy(extra: ParamMap): Estimator[AutoencoderModel] = defaultCopy(extra)

  /**
   * :: DeveloperApi ::
   *
   * Derives the output schema from the input schema.
   */
  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    require(inputType.isInstanceOf[VectorUDT],
      s"Input column ${$(inputCol)} must be a vector column")
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }
}
/**
 * :: Experimental ::
 * Autoencoder model.
 *
 * @param layers array of layer sizes including input and output
 * @param weights weights (or parameters) of the model
 */
@Experimental
class AutoencoderModel private[ml] (override val uid: String,
                                    layers: Array[Int],
                                    weights: Vector)
  extends Model[AutoencoderModel] with AutoencoderParams {

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  // TODO: make sure that the same topology is created as in Autoencoder
  private val autoecoderModel =
    FeedForwardTopology.multiLayerPerceptron(layers, true).getInstance(weights)


  override def copy(extra: ParamMap): AutoencoderModel = {
    copyValues(new AutoencoderModel(uid, layers, weights), extra)
  }

  /**
   * Transforms the input dataset.
   */
  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val pcaOp = udf { autoecoderModel.predict _ }
    dataset.withColumn($(outputCol), pcaOp(col($(inputCol))))
  }

  /**
   * :: DeveloperApi ::
   *
   * Derives the output schema from the input schema.
   */
  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    require(inputType.isInstanceOf[VectorUDT],
      s"Input column ${$(inputCol)} must be a vector column")
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }
}
