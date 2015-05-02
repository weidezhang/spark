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

import scala.util.{Success, Failure}

import org.apache.spark.Logging
import org.apache.spark.ps.VectorClock
import org.apache.spark.ps.local.LocalPSMessage._
import org.apache.spark.rpc.{ThreadSafeRpcEndpoint, RpcEndpointRef, RpcCallContext, RpcEnv}
import scala.concurrent.ExecutionContext.Implicits.global



import scala.collection.mutable

class LocalPSServer(override val rpcEnv: RpcEnv, serverId: Int)
  extends ThreadSafeRpcEndpoint with Logging  {
  private val row = Array(0.0)
  private val ROW_ID = 0
  private val clients = mutable.HashMap.empty[Int, (String, RpcEndpointRef)]
  private val vectorClock = new VectorClock
  private val pendingClients = mutable.Set.empty[Int]


  override def receiveAndReply(context: RpcCallContext)
  : PartialFunction[Any, Unit] = {
    case ConnectServer =>
      context.reply(ServerConnected(serverId))

    case RegisterClient(clientId, url) =>
      if (!clients.contains(clientId)) {
        vectorClock.addClock(clientId, 0)
      }
      logDebug(s"get registered from client: $clientId, $url, " +
        LocalPSMaster.getUriByRef(rpcEnv, context.sender))
      clients += clientId -> (url, context.sender)
      context.reply(ClientRegistered(vectorClock(clientId)))

    case RowRequest(clientId, rowId, clock) =>
      val minClock = vectorClock.getMinClock()
      logDebug(s"server $serverId get row request: $clientId $rowId $clock")
      context.reply(true)
      logDebug(s"server $serverId has reply row request: $clientId $rowId $clock")


      if (minClock >= clock) {
        // workaround for don't have send with reply method,
        // must call context.reply(true) first, otherwise this row reply is ignored
        logDebug(s"server $serverId can reply this " +
          s"row request directly: $clientId $rowId $clock")
        replyRow(clientId, RowRequestReply(clientId, ROW_ID, minClock, row))
      } else {
        pendingClients += clientId
      }

  }

  override def receive: PartialFunction[Any, Unit] = {
    case UpdateRow(clientId, rowId, clock, rowDelta) =>
      logDebug(s"server $serverId received update: $clientId, $rowId, $clock " +
        rowDelta.mkString(" "))
      row(0) += rowDelta(0)

    case Clock(clientId, clock) =>
      logDebug(s"server $serverId received clock: $clientId $clock")
      val minClock = vectorClock.tickUntil(clientId, clock)
      // if min clock has changed(return no zero value)
      if (minClock != 0) {
        logDebug(s"server $serverId can reply row requests " +
          s"after clock change: $clientId $clock")
        pendingClients.foreach { clientId =>
          replyRow(clientId, RowRequestReply(clientId, ROW_ID, minClock, row))
        }
        pendingClients.clear()
      }
  }

  private def replyRow(clientId: Int, reply: RowRequestReply) = {
    require(clients.contains(clientId), s"must contain $clientId")
    rpcEnv.asyncSetupEndpointRefByURI(clients(clientId)._1) onComplete {
      case Success(ref) =>
        ref.send(reply)
      case Failure(e) =>
        logError(s"server $serverId can't get client ref: $clientId", e)
    }
    clients(clientId)._2.send(reply)
  }
}
