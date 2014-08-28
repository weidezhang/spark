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

import org.apache.spark.mllib.evaluation.MulticlassMetrics

import scala.math.random
import org.scalatest.FunSuite
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.{MLUtils, LocalSparkContext}



class NeuralNetworkSuite extends FunSuite with LocalSparkContext {

  /* training set */
  val inputs = Array[Array[Double]](
    Array[Double](0,0),
    Array[Double](0,1),
    Array[Double](1,0),
    Array[Double](1,1)
  )
  val outputs = Array[Double](0, 1, 1, 0)
  /* NN */
  val inputSize = 2
  val hiddenSize = 5
  val outputSize = 1
  val data = inputs.zip(outputs).map{ case(features, label) =>
    new LabeledPoint(label, Vectors.dense(features))}


  test("XOR classification") {
    val rddData = sc.parallelize(data, 2)
    val predictor = NeuralNetwork.train(rddData, Array(hiddenSize), 1000, 0.7)
    val predictionAndLabels = rddData.map(lp => (predictor.predict(lp.features), lp.label)).collect()
    predictionAndLabels.foreach(x => assert(x._1 == x._2))
  }
}
