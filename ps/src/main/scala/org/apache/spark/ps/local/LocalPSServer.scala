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

import org.apache.spark.ps.VectorClock
import org.apache.spark.ps.local.LocalPSMessage._
import org.apache.spark.rpc.{RpcEndpointRef, RpcCallContext, RpcEnv}


import scala.collection.mutable

class LocalPSServer(override val rpcEnv: RpcEnv, serverId: Int)
  extends LoggingRpcEndpoint {
  private val row = Array(0.0)
  private val ROW_ID = 0
  private val clientIds = mutable.Set.empty[Int]
  private val vectorClock = new VectorClock
  private val pendingClients = mutable.HashMap.empty[Int, RpcEndpointRef]


  override def receiveAndReplyWithLog(context: RpcCallContext)
  : PartialFunction[LocalPSMessage, Unit] = {
    case ConnectServer =>
      context.reply(ServerConnected(serverId))

    case RegisterClient(clientId) =>
      if (!clientIds.contains(clientId)) {
        vectorClock.addClock(clientId, 0)
      }
      clientIds += clientId
      context.reply(ClientRegistered(vectorClock(clientId)))

    case RowRequest(clientId, rowId, clock) =>
      val minClock = vectorClock.getMinClock()
      if (minClock >= clock) {
        context.reply(RowRequestReply(clientId, ROW_ID, minClock, row))
      } else {
        pendingClients(clientId) = context.sender
      }

    case Clock(clientId, clock) =>
      val minClock = vectorClock.tickUntil(clientId, clock)
      // if min clock has changed(return no zero value)
      if (minClock != 0) {
        pendingClients.foreach { case (cid, clientRef) =>
          clientRef.send(RowRequestReply(cid, ROW_ID, minClock, row))
        }
      }

    case UpdateRow(clientId, rowId, clock, rowDelta) =>
      row(0) += rowDelta(0)

  }
}
