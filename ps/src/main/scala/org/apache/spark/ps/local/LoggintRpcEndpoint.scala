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

import org.apache.spark.{SparkException, Logging}
import org.apache.spark.rpc.{RpcCallContext, ThreadSafeRpcEndpoint}

trait LoggingRpcEndpoint extends ThreadSafeRpcEndpoint with Logging{
  override def receiveAndReply(context: RpcCallContext): PartialFunction[Any, Unit] = {
    case m: LocalPSMessage =>
      logInfo("receive message: " + m)
      receiveAndReplyWithLog(context)(m)
  }

  def receiveAndReplyWithLog(context: RpcCallContext): PartialFunction[LocalPSMessage, Unit] = {
    case _ => context.sendFailure(new SparkException(self + " won't reply anything"))
  }

  override def receive: PartialFunction[Any, Unit] = {
    case m: LocalPSMessage =>
      logInfo("receive message: " + m)
      receiveWithLog(m)
  }

  def receiveWithLog: PartialFunction[LocalPSMessage, Unit] = {
    case _ => throw new SparkException(self + " does not implement 'receive'")
  }
}
