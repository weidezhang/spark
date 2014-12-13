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

package org.apache.spark.mllib.neuralNetwork

import scala.collection.JavaConversions._

import org.scalatest.{Matchers, FunSuite}
import org.apache.spark.mllib.util.MinstDatasetSuite


class MLPSuite extends FunSuite with MinstDatasetSuite with Matchers {

  ignore("MLP") {
    val (data, numVisible) = minstTrainDataset(2500)
    data.cache()
    val nn = MLP.train(data, 20, 1000, Array(numVisible, 500, 10), 0.05, 0.0, 0.01)
    // val nn = MLP.runLBFGS(data, Array(numVisible, 1000, 10), 100, 4000, 1e-12, 0.0)
    MLP.runSGD(data, nn, 37, 6000, 0.1, 0.0, 0.01)

    val (dataTest, _) = minstTrainDataset(10000, 5000)
    println("Error: " + MLP.error(dataTest, nn, 100))
  }
}
