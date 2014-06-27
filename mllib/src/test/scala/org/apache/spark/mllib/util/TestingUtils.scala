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

package org.apache.spark.mllib.util

import scala.io.Source._

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression._

object TestingUtils  {

  implicit class DoubleWithAlmostEquals(val x: Double) {
    // An improved version of AlmostEquals would always divide by the larger number.
    // This will avoid the problem of diving by zero.
    def almostEquals(y: Double, epsilon: Double = 1E-10): Boolean = {
      if(x == y) {
        true
      } else if(math.abs(x) > math.abs(y)) {
        math.abs(x - y) / math.abs(x) < epsilon
      } else {
        math.abs(x - y) / math.abs(y) < epsilon
      }
    }
  }

  implicit class VectorWithAlmostEquals(val x: Vector) {
    def almostEquals(y: Vector, epsilon: Double = 1E-10): Boolean = {
      x.toArray.corresponds(y.toArray) {
        _.almostEquals(_, epsilon)
      }
    }
  }

  def validateCategoricalPrediction(
      predictions: Seq[Double],
      input: Seq[LabeledPoint],
      acc: Double): Boolean = {
    val numOffPredictions = predictions.zip(input).count { case (prediction, expected) =>
      prediction.round != expected.label.round
    }
    ((input.length - numOffPredictions).toDouble / input.length) > acc
  }

  def loadIrisDataSet: Seq[LabeledPoint] = {
    val file = this.getClass.getResource(
      "/org/apache/spark/mllib/dataset/classification/iris.csv").toURI
    val lines = fromFile(file).getLines.drop(1)

    lines.map(line => {
      val temp = line.toString.split(",")
      val y: Int = temp(4) match {
        case "Iris-setosa" => 0
        case "Iris-versicolor" => 1
        case "Iris-virginica" => 2
      }
      val x = temp.slice(0, 4).map(_.toDouble)
      LabeledPoint(y, Vectors.dense(x))
    }).toSeq
  }
}
