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

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{SparkEnv, Logging, SparkContext}
import org.apache.spark.ps.{PSMasterInfo, PSMaster}
import org.apache.spark.ps.local.LocalPSMessage._
import org.apache.spark.rpc.{ThreadSafeRpcEndpoint, RpcCallContext, RpcEndpointRef, RpcEnv}


/**
 * start master and server
 */
class LocalPSMaster(
  sc: SparkContext,
  config: LocalPSConfig) extends PSMaster with Logging{

  private var masterRef: Option[RpcEndpointRef] = None
  private val serverRefs =  ArrayBuffer[RpcEndpointRef]()
  private val rpcEnv = sc.env.rpcEnv
  private var ready = false

  def start(): Unit = {
    logInfo("Start local servers")
    for (i <- 0 until config.serverNum) {
      val serverRef =
        rpcEnv.setupEndpoint(s"PSServer_$i", new LocalPSServer(rpcEnv, i, config.rowSize))
      serverRefs += serverRef
    }

    logInfo("Start local master")
    masterRef = Some(rpcEnv.setupEndpoint("PSMaster",
      new LocalPSMasterEndpoint(rpcEnv, serverRefs.toArray, this)))
    while (!isReady) {
      Thread.sleep(100)
    }
  }

  def masterInfo: PSMasterInfo = {
    require(isReady, "can't get master url before master get ready")
    LocalPSMasterInfo(LocalPSMaster.getUriByRef(rpcEnv, masterRef.get), config.rowNum)
  }

  def isReady: Boolean = ready

  def stop(): Unit = {
    serverRefs.foreach(rpcEnv.stop)
    masterRef.foreach(rpcEnv.stop)
  }

  class LocalPSMasterEndpoint(
    override val rpcEnv: RpcEnv,
    serverRefs: Array[RpcEndpointRef],
    master: LocalPSMaster
    ) extends ThreadSafeRpcEndpoint with Logging  {
    private val serverReady = serverRefs.map(_ => false)
    private val serverUrls = serverRefs.map(LocalPSMaster.getUriByRef(rpcEnv, _))


    override def onStart(): Unit = {
      serverRefs.foreach { serverRef =>
        val serverConnected = serverRef.askWithRetry[ServerConnected](ConnectServer)
        serverReady(serverConnected.serverId) = true
        setReady()
      }
    }

    override def receiveAndReply(context: RpcCallContext)
    : PartialFunction[Any, Unit] = {
      case _: RegisterClient =>
        context.reply(ServerUrls(serverUrls))
    }

    private def setReady(): Unit = {
      if (serverReady.forall(b => b)) {
        ready = true
      }
    }
  }

}

object LocalPSMaster {
  val actorSystemName = SparkEnv.driverActorSystemName

  def getUriByRef(rpcEnv: RpcEnv, rpcEndpointRef: RpcEndpointRef): String = {
    rpcEnv.uriOf(actorSystemName, rpcEndpointRef.address, rpcEndpointRef.name)
  }
}
