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
import scala.concurrent.ExecutionContext.Implicits.global

import org.apache.spark.SparkEnv
import org.apache.spark.Logging
import org.apache.spark.ps.PSClient
import org.apache.spark.rpc.{ThreadSafeRpcEndpoint, RpcEndpointRef, RpcEnv}
import org.apache.spark.ps.local.LocalPSMessage._



class LocalPSClient(val clientId: Int, val masterUrl: String) extends PSClient{

  private var initialized = false
  private val rows = mutable.HashMap.empty[Int, Array[Double]]
  private var clientEndpoint: LocalPSClientEndpoint = null
  private var currentClock: Int = 0
  private val rpcEnv = SparkEnv.get.rpcEnv
  private var clientEndpointRef: Option[RpcEndpointRef] = None

  init()

  def init(): Unit = {
    clientEndpoint = new LocalPSClientEndpoint(
      rpcEnv, masterUrl)
    val ref = rpcEnv.setupEndpoint(s"PSClient_$clientId", clientEndpoint)
    clientEndpointRef = Some(ref)
    while (!initialized) {
      Thread.sleep(100)
    }
  }

  /** get parameter indexed by key from parameter server
    */
  def get(rowId: Int): Array[Double] = {
    if (rows.contains(rowId)) {
      rows.remove(rowId)
    }

    clientEndpoint.rowRequest(rowId, currentClock)

    while (!rows.contains(rowId)) {
      Thread.sleep(100)
    }

    rows(rowId)
  }

  //  // get multiple parameters from parameter server
  //  def multiGet[T](keys: Array[String]): Array[T]

  /** add parameter indexed by `key` by `delta`
    *  if multiple `delta` to update on the same parameter
    *  use `reduceFunc` to reduce these `delta`s frist
    */
  def update(rowId: Int, delta: Array[Double]): Unit = {
    clientEndpoint.updateRow(rowId, currentClock, delta)
  }

  //  // update multiple parameters at the same time, use the same `reduceFunc`.
  //  def multiUpdate(keys: Array[String], delta: Array[T], reduceFunc: (T, T) => T: Unit

  /** advance clock to indicate that current iteration is finished.
    */
  def clock(): Unit = {
    currentClock += 1
    clientEndpoint.clock(currentClock)
  }

  def stop(): Unit = {
    clientEndpointRef.foreach(rpcEnv.stop)
  }


  class LocalPSClientEndpoint(
    override val rpcEnv: RpcEnv,
    masterUrl: String
    ) extends ThreadSafeRpcEndpoint with Logging {

    // TODO: need to consider use String(url) or RpcEndpointRef
    private var servers: Option[Array[RpcEndpointRef]] = None

    var master: Option[RpcEndpointRef] = None

    override def onStart(): Unit = {
      logInfo("PSClient connecting to : " + masterUrl)
      val url = LocalPSMaster.getUriByRef(rpcEnv, self)
      rpcEnv.asyncSetupEndpointRefByURI(masterUrl) onComplete  {
        case Success(ref) =>
          master = Some(ref)
          val urls = ref.askWithRetry[ServerUrls](RegisterClient(clientId, url)).urls
          Future.sequence(urls.iterator.map(rpcEnv.asyncSetupEndpointRefByURI)) onComplete  {
            case Success(refs) =>
              servers = Some(refs.toArray)
              val clocks = servers.get.map { serverRef =>
                serverRef.askWithRetry[ClientRegistered](RegisterClient(clientId, url))
              }
              val maxClock = clocks.maxBy(r => r.clock).clock
              if (maxClock > currentClock) {
                logInfo("set client's clock to max clock returned by server" +
                  s"currentClock: $currentClock, maxClock: $maxClock")
                currentClock = maxClock
              }

              initialized = true

            case Failure(e) => logError(s"Client $clientId cannot get server refs: " + urls.mkString(", "), e)
          }

        case Failure(e) => logError(s"Cannot register with driver: $masterUrl", e)
      }
    }

    override def receive: PartialFunction[Any, Unit] = {
      case RowRequestReply(cid, rowId, clock, rowData) =>
        require(cid == clientId,
          s"send rowData to wrong client: $cid vs $clientId")
        logDebug(s"Client $clientId get reply from server: $rowId, " + rowData.mkString(" "))
        rows(rowId) = rowData
    }

    // TODO: use other hash functions if necessary
    private def serverByRowId(rowId: Int): RpcEndpointRef = {
      require(servers.isDefined, "must initialized first before row request")
      val totalRows = servers.get.length
      val serverIndex = rowId % totalRows
      servers.get(serverIndex)
    }

    def updateRow(rowId: Int, clock: Int, rowDelta: Array[Double]): Unit = {
      serverByRowId(rowId).send(UpdateRow(clientId, rowId, clock, rowDelta))
    }

    def rowRequest(rowId: Int, clock: Int): Unit = {
      serverByRowId(rowId).ask[Boolean](RowRequest(clientId, rowId, clock)) onComplete {
        case Success(b) =>
          logDebug(s"Client $clientId: row request returns $b")
        case Failure(e) => logError(s"Client $clientId cannot get row request", e)
      }
    }

    def clock(clock: Int): Unit = {
      require(servers.isDefined, "must initialized first before clock")
      servers.get.foreach { server =>
        server.send(Clock(clientId, clock))
      }
    }
  }
}
