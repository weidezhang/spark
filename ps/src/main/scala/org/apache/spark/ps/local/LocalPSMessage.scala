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

private[local] sealed trait LocalPSMessage extends Serializable

private[local] object LocalPSMessage {
  case class RegisterClient(clientId: Int, url: String) extends LocalPSMessage
  case class ServerUrls(urls: Array[String]) extends LocalPSMessage


  // TODO: add tableId
  case class ClientRegistered(clock: Int) extends LocalPSMessage
  case class RowRequest(clientId: Int, rowId: Int, clock: Int) extends LocalPSMessage
  // TODO: change rowData to generic type
  case class RowRequestReply(clientId: Int, rowId: Int, clock: Int, rowData: Array[Double])
    extends LocalPSMessage
  case class UpdateRow(clientId: Int, rowId: Int, clock: Int, rowDelta: Array[Double])
    extends LocalPSMessage
  case class Clock(clientId: Int, clock: Int) extends LocalPSMessage


  case object ConnectServer extends LocalPSMessage
  case class ServerConnected(serverId: Int) extends LocalPSMessage
}
