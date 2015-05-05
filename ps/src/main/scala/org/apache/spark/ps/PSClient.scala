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

import org.apache.spark.ps.local.{LocalPSClient, LocalPSMasterInfo}

trait PSClient {
  def init(): Unit

  /** get parameter indexed by key from parameter server
    */
  def get(rowId: Int): Array[Double]

//  // get multiple parameters from parameter server
//  def multiGet[T](keys: Array[String]): Array[T]

  // add parameter indexed by `key` by `delta`,
  // if multiple `delta` to update on the same parameter,
  // use `reduceFunc` to reduce these `delta`s frist.


  /** add parameter indexed by `key` by `delta`
   *  if multiple `delta` to update on the same parameter
   *  use `reduceFunc` to reduce these `delta`s frist
   */
  def update(rowId: Int, delta: Array[Double]): Unit

//  // update multiple parameters at the same time, use the same `reduceFunc`.
//  def multiUpdate(keys: Array[String], delta: Array[T], reduceFunc: (T, T) => T: Unit

  /** advance clock to indicate that current iteration is finished.
    */
  def clock(): Unit
}


object PSClient {
  def apply(clientId: Int, masterInfo: PSMasterInfo): PSClient = masterInfo match {
    case info: LocalPSMasterInfo => new LocalPSClient(clientId, info.masterUrl)
  }
}
