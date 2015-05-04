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

import com.sun.javaws.exceptions.InvalidArgumentException
import org.apache.spark.SparkContext
import org.apache.spark.ps.local.{LocalPSConfig, LocalPSMaster}
import org.apache.spark.rdd.RDD

// TODO: initialized parameters
class PSContext(sc: SparkContext, config: PSConfig) {
  var psMaster: PSMaster = config match {
    case _: LocalPSConfig =>
      new LocalPSMaster(config)
    case _ =>
      throw new InvalidArgumentException("Unknown PS Config")
  }

  def start(): Unit = {
    psMaster.start()
  }

  def stop(): Unit = {
    psMaster.stop()
  }

  def masterUrl: String = psMaster.masterUrl
}


abstract class PSContext(sc: SparkContext, config: PSConfig) {
  protected val psMaster: PSMaster

  def start(): Unit

  def uploadParams(initialParams: Array[Array[Double]]): Unit
  def downloadParams(): Array[Array[Double]]

  def loadParams(path: String, numPartitions: Int): Unit
  def saveParams(path: String): Unit
}
