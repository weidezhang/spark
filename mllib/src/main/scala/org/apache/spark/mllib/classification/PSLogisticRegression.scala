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

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ps.{TableInfo, PSContext}
import org.apache.spark.ps.local.LocalPSClient
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.BernoulliSampler

object PSLogisticRegression {
  def train(
    sc: SparkContext,
    input: RDD[LabeledPoint],
    numIterations: Int,
    stepSize: Double,
    miniBatchFraction: Double): LogisticRegressionModel = {
    val numFeatures = input.map(_.features.size).first()

    val psContext = new PSContext(sc)
    psContext.start(TableInfo(numFeatures))
    val masterUrl = psContext.masterUrl

    val weights: RDD[Array[Double]] = input.mapPartitionsWithIndex { (index, iter) =>
      val arr = iter.toArray
      val client = new LocalPSClient(index, masterUrl)
      val sampler = new BernoulliSampler[LabeledPoint](miniBatchFraction)

      for (i <- 0 to numIterations) {
        val w = Vectors.dense(client.get(0))
        val delta = Vectors.dense(new Array[Double](numFeatures))

        sampler.setSeed(i + 42)
        sampler.sample(arr.toIterator).foreach { point =>
          val weights = w.copy
          axpy(1.0, delta, weights)
          val data = point.features
          val label = point.label
          val margin = -1.0 * dot(data, weights)
          val multiplier = (1.0 / (1.0 + math.exp(margin))) - label
          axpy((-1) * stepSize / math.sqrt(i + 1) * multiplier, data, delta)
        }
        client.update(0, delta.toArray)
        client.clock()
      }

      Iterator(client.get(0))
    }

    val w = Vectors.dense(weights.first())
    val intercept = 0.0

    new LogisticRegressionModel(w, intercept).clearThreshold()
  }
}
