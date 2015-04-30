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

package org.apache.spark.ps.local


import scala.util.{Failure, Success}
import scala.concurrent.Future
import scala.collection.mutable

import org.apache.spark.{Logging, SparkEnv}
import org.apache.spark.ps.PSClient
import org.apache.spark.rpc.{RpcEndpointRef, ThreadSafeRpcEndpoint, RpcEnv}
import LocalPSMessage._

class LocalPSClientEndpoint(
  override val rpcEnv: RpcEnv,
  client: LocalPSClient,
  masterUrl: String
  ) extends ThreadSafeRpcEndpoint with Logging {

  private var servers: Option[Array[RpcEndpointRef]] = None
  var master: Option[RpcEndpointRef] = None

  override def onStart(): Unit = {
    import scala.concurrent.ExecutionContext.Implicits.global
    logInfo("PSClient connecting to : " + master)
    rpcEnv.asyncSetupEndpointRefByURI(masterUrl) onComplete  {
      case Success(ref) => {
        master = Some(ref)
        ref.send(RegisterClient(client.clientId))
      }
      case Failure(e) => logError(s"Cannot register with driver: $masterUrl", e)
    }
  }

  // TODO: avoid send message twice
  override def receive: PartialFunction[Any, Unit] = {
    case ServerUrls(urls) => {
      Future.sequence(urls.map(rpcEnv.asyncSetupEndpointRefByURI)) onComplete  {
        case Success(refs) => {
          servers = Some(refs)
          client.setInitialized()
        }
        case Failure(e) => logError(s"Cannot get server refs: " + urls.mkString(", "), e)
      }
    }
    case RowRequestReply(clientId, rowId, clock, rowData) => {
      require(clientId == client.clientId,
        s"send rowData to wrong client: $clientId vs ${client.clientId}")
      client.rows(rowId) = rowData
    }


    // redirect message to server
    // TODO: find a way to avoid this, call functions directly
    case UpdateRow(clientId, rowId, clock, rowDelta) => {
      serverByRowId(rowId).send(UpdateRow(clientId, rowId, clock, rowDelta))
    }
    case RowRequest(clientId, rowId, clock) => {
      serverByRowId(rowId).send(RowRequest(clientId, rowId, clock))
    }
    case Clock(clock) => {
      clockAllServers(clock)
    }
  }

  // TODO: use other hash functions if neccessary
  private def serverByRowId(rowId: Int): RpcEndpointRef = {
    require(servers.isDefined, "must initialized first before row request")
    val totalRows = servers.get.length
    val serverIndex = rowId % totalRows
    servers.get(serverIndex)
  }

  private def clockAllServers(clock: Int)  {
    require(servers.isDefined, "must initialized first before clock")
    servers.get.foreach { server =>
      server.send(Clock(clock))
    }
  }
}

class LocalPSClient(val clientId: Int, val masterUrl: String) extends PSClient{
  override type T = Array[Double]

  private var initialized = false
  val rows = mutable.HashMap.empty[Int, Array[Double]]
  var endpoint: RpcEndpointRef = null
  var currentClock: Int = 0


  def init(): Unit = {
    val rpcEnv = SparkEnv.get.rpcEnv
    endpoint = rpcEnv.setupEndpoint("PSClient", new LocalPSClientEndpoint(
      rpcEnv, this, masterUrl))
    while (!initialized) {
      this.wait(100)
    }
  }

  /** get parameter indexed by key from parameter server
    */
  def get(rowId: Int): T = {
    if (rows.contains(rowId)) {
      rows.remove(rowId)
    }

    endpoint.send(RowRequest(clientId, rowId, currentClock))

    while (!rows.contains(rowId)) {
      this.wait(100)
    }

    rows(rowId)
  }

  //  // get multiple parameters from parameter server
  //  def multiGet[T](keys: Array[String]): Array[T]

  /** add parameter indexed by `key` by `delta`
    *  if multiple `delta` to update on the same parameter
    *  use `reduceFunc` to reduce these `delta`s frist
    */
  def update(rowId: Int, delta: T): Unit = {
    endpoint.send(UpdateRow(clientId, rowId, currentClock, delta))
  }

  //  // update multiple parameters at the same time, use the same `reduceFunc`.
  //  def multiUpdate(keys: Array[String], delta: Array[T], reduceFunc: (T, T) => T: Unit

  /** advance clock to indicate that current iteration is finished.
    */
  def clock(): Unit = {
    currentClock += 1
    endpoint.send(Clock(currentClock))
  }

  def setInitialized(): Unit = {
    initialized = true
  }


}
