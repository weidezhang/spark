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

import org.apache.spark.{SparkEnv, Logging, SparkContext}
import org.apache.spark.ps.PSMaster
import org.apache.spark.ps.local.LocalPSMessage._
import org.apache.spark.rpc.{ThreadSafeRpcEndpoint, RpcCallContext, RpcEndpointRef, RpcEnv}

import scala.collection.mutable.ArrayBuffer
import scala.util.{Success, Failure}


class LocalPSMasterEndpoint(
  override val rpcEnv: RpcEnv,
  serverUrls: Array[String],
  master: LocalPSMaster
  ) extends ThreadSafeRpcEndpoint with Logging  {
  private val serverReady = serverUrls.map(_ => false)


  override def onStart(): Unit = {
    import scala.concurrent.ExecutionContext.Implicits.global
    serverUrls.foreach { serverUrl =>
      rpcEnv.asyncSetupEndpointRefByURI(serverUrl) onComplete {
        case Success(serverRef) =>
          val serverConnected = serverRef.askWithRetry[ServerConnected](ConnectServer)
          serverReady(serverConnected.serverId) = true
          setReady()
        case Failure(e) => logError("Cannot get server by url: " + serverUrl, e)
      }
    }
  }

  override def receiveAndReply(context: RpcCallContext)
  : PartialFunction[Any, Unit] = {
    case _: RegisterClient =>
      context.reply(ServerUrls(serverUrls))
  }

  private def setReady(): Unit = {
    if (serverReady.forall(b => b)) {
      master.setReady()
//      master.notify()
    }
  }
}

/**
 * start master and server
 */
class LocalPSMaster(
  sc: SparkContext,
  numServers: Int) extends PSMaster with Logging{

  private var masterRef: Option[RpcEndpointRef] = None
  private val serverRefs =  ArrayBuffer[String]()
  private val rpcEnv = sc.env.rpcEnv
  private var ready = false

  def start(): Unit = {
    logInfo("Start local servers")
    for (i <- 0 until numServers) {
      val serverRef = rpcEnv.setupEndpoint(s"PSServer_$i", new LocalPSServer(rpcEnv, i))
      serverRefs += LocalPSMaster.getUriByRef(rpcEnv, serverRef)
    }

    logInfo("Start local master")
    masterRef = Some(rpcEnv.setupEndpoint("PSMaster",
      new LocalPSMasterEndpoint(rpcEnv, serverRefs.toArray, this)))
    while (!isReady) {
      Thread.sleep(100)
    }
  }

  def masterUrl: String = {
    require(isReady, "can't get master url before master get ready")
    LocalPSMaster.getUriByRef(rpcEnv, masterRef.get)
  }

  def isReady: Boolean = ready

  def setReady(): Unit = {
    ready = true
  }


  def stop(): Unit = {

  }


}

object LocalPSMaster {
  private val actorSystemName = SparkEnv.driverActorSystemName

  def getUriByRef(rpcEnv: RpcEnv, rpcEndpointRef: RpcEndpointRef): String = {
    rpcEnv.uriOf(actorSystemName, rpcEndpointRef.address, rpcEndpointRef.name)
  }
}
