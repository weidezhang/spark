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

class PSContext(sc: SparkContext, config: PSConfig) {
  private var initialized = false
  var psMaster: PSMaster = config match {
    case c: LocalPSConfig =>
      new LocalPSMaster(sc, c)
    case _ =>
      throw new IllegalArgumentException("Unknown PS Config")
  }

  def start(): Unit = {
    psMaster.start()
    initialized = true
  }

  def stop(): Unit = {
    psMaster.stop()
  }

  def runPSJob[T: ClassTag, U: ClassTag](rdd: RDD[T])
    (func: (Int, Array[T], PSClient) => Iterator[U]): RDD[U] = {
    require(initialized, "must call PSContext.start() to initialize before runPSJob")
    val masterInfo = psMaster.masterInfo
    rdd.mapPartitionsWithIndex { (pid, iter) =>
      val arr = iter.toArray
      val client = PSClient(pid, masterInfo)
      val result = func(pid, arr, client)
      client.stop()
      result
    }
  }

//  def uploadParams(initialParams: Array[Array[Double]]): Unit = {
//
//  }
//
//  def downloadParams(): Array[Array[Double]] = {
//
//  }
//
//  def loadParams(path: String, numPartitions: Int): Unit = {
//
//  }
//
//  def saveParams(path: String): Unit = {
//
//  }
}
