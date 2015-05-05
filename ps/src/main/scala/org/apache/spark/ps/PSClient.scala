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

import org.apache.spark.{SparkProcessEnv, SparkEnv}
import org.apache.spark.ps.local.{LocalPSClient, LocalPSMasterInfo}

trait PSClient {
  protected val processEnv = SparkEnv.get.initProcessEnv(initProcessEnv)

  /**
   * Initialize a process environment in SparkEnv
   * to enable clients perform some actions in a process level.
   * Implementations that need process level actions
   * can inherit {{SparkProcessEnv}} and
   * let this function return a object of the derived class.
   * Process environment will only be initialized once.
   * If it has been initialized, nothing will be done if this function gets called.
   * @return the process env initialized
   */
  def initProcessEnv(): SparkProcessEnv

  /**
   * Get parameter indexed by key from parameter server
   */
  def get(rowId: Int): Array[Double]

  /**
   * Add parameter indexed by `key` by `delta`.
   */
  def update(rowId: Int, delta: Array[Double]): Unit

  /**
   * Advance clock to indicate that current iteration is finished.
   */
  def clock(): Unit

  /**
   * Number of rows
   */
  def rowNum: Int

  /**
   * stop parameter server client
   */
  def stop(): Unit
}


object PSClient {
  /**
   * Construct a new parameter client based on clientId and masterInfo.
   * New implementations of parameter server needs to add an case in this function
   * to construct specific parameter server client.
   * @param clientId: id of parameter client, corresponding to the partition id
   * @param masterInfo: master information. Use masterInfo to establish connection with master.
   * @return a parameter server client
   */
  def apply(clientId: Int, masterInfo: PSMasterInfo): PSClient = masterInfo match {
    case info: LocalPSMasterInfo => new LocalPSClient(clientId, info.masterUrl, info.rowNum)
  }
}
