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

import org.scalatest.{Matchers, FunSuite}
import org.apache.spark.mllib.util.{MinstDatasetSuite}

class DBNSuite extends FunSuite with MinstDatasetSuite with Matchers {

  ignore("DBN") {
    val (data, numVisible) = minstTrainDataset(2500)
    val dbn = DBN.initializeDBN(Array(numVisible, 500, 10))
    DBN.pretrain(data, 10, 1000, dbn, 0.05, 0.6, 0.0, 0.01)
    DBN.finetune(data, 10, 2000, dbn, 0.05, 0.0, 0.01)
    val (dataTest, _) = minstTrainDataset(5000, 2500)
    println("Error: " + MLP.error(dataTest, dbn.nn, 100))

  }

}
