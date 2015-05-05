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

package org.apache.spark.ps

import org.apache.spark.SparkContext
import org.apache.spark.ps.local.{LocalPSConfig, LocalPSMaster}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Main entry point of the parameter server functionality.
 * A PSContext can be used to run parameter server job, initialize parameters and fetch parameters.
 * @param sc: SparkContext
 * @param config: parameter server configuration
 */
class PSContext(sc: SparkContext, config: PSConfig) {
  private var initialized = false
  private val psMaster: PSMaster = config match {
    case c: LocalPSConfig =>
      new LocalPSMaster(sc, c)
    case _ =>
      throw new IllegalArgumentException("Unknown PS Config")
  }

  /**
   * Start parameter server context, must call this function before any other functions.
   */
  def start(): Unit = {
    psMaster.start()
    initialized = true
  }

  /**
   * Stop parameter server context if there is no need for parameter server.
   */
  def stop(): Unit = {
    psMaster.stop()
  }

  /**
   * Run Spark job on the input rdd with parameter server.
   * User specified a `func` to run on each partition of the input rdd.
   * Three arguments are passed to `func`: partitionId, data of input in this partition as an array
   * and a PSClient. This `func` can get parameters from parameter server through PSClient,
   * compute delta to update the parameters based on the current parameters and data of this partition
   * and update parameters using PSClient.
   * This `func` return a new iterator of type `U` to construct a new rdd.
   * See {{org.apache.spark.mllib.classification.PSLogisticRegression}} for an example.
   * @param rdd: input rdd
   * @param func: function to run. (partitionId, data, PSClient) are passed as arguments.
   * @tparam T type of input rdd
   * @tparam U type of iterator that `func` returns
   * @return a new rdd that is constructed by concatenate all the iterators
   *         of each partition returned by `func`.
   *
   * Note: this function is evaluated lazily when action is done on the return rdd.
   */
  def runPSJob[T: ClassTag, U: ClassTag](rdd: RDD[T])
    (func: (Int, Array[T], PSClient) => Iterator[U]): RDD[U] = {
    require(initialized, "Must call PSContext.start() to initialize before runPSJob")
    val masterInfo = psMaster.masterInfo
    rdd.mapPartitionsWithIndex { (pid, iter) =>
      val arr = iter.toArray
      val client = PSClient(pid, masterInfo)
      val result = func(pid, arr, client)
      client.stop()
      result
    }
  }

  /**
   * Upload an array of parameters to parameter server.
   * Rows are corresponding to initialParams. e.g. row(0) is corresponding to initialParams(0).
   * @param initialParams: initialParams to upload.
   */
  def uploadParams(initialParams: Array[Array[Double]]): Unit = {
    runPSJob(sc.parallelize(initialParams, initialParams.length)) { (idx, arr, client) =>
      client.update(idx, arr(0))
      Iterator()
    }.count()

    ()
  }

  /**
   * Download parameters from parameter server to driver.
   * @return parameters as an array.
   */
  def downloadParams(): Array[Array[Double]] = {
    runPSJob(sc.parallelize(Array(1), 1)) { (_, _, client) =>
      Array.tabulate(client.rowNum)(client.get).iterator
    }.collect()
  }

  /**
   * Load model from a RDD.
   * @param rdd input rdd
   */
  def loadParams(rdd: RDD[Array[Double]]): Unit = {
    runPSJob(rdd.zipWithIndex()) { (_, arr, client) =>
      arr.foreach { case (row, idx) =>
        client.update(idx.toInt, row)
      }
      Iterator()
    }.count()

    ()
  }

  /**
   * Save model to a RDD
   * @param numPartitions number of partitions of the result rdd
   */
  def saveParams(numPartitions: Int): RDD[Array[Double]] = {
    runPSJob(sc.parallelize(0 until config.rowNum, numPartitions)) { (_, arr, client) =>
      arr.map { idx =>
        client.get(idx)
      }.iterator
    }
  }
}
