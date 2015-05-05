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

import scala.collection.mutable

import org.apache.spark.Logging

class VectorClock extends Logging {
  private var minClock = -1
  private val id2clock = mutable.HashMap.empty[Int, Int]

  def this(ids: Array[Int]) = {
    this()
    minClock = 0
    ids.foreach { id =>
      id2clock(id) = 0
    }
  }

  def addClock(id: Int, clock: Int): Unit = {
    id2clock(id) = clock
    if (minClock == -1 || clock < minClock) {
      minClock = clock
    }
    logDebug(s"add clock: ($id, $clock), min clock is :$minClock")
  }

  def tick(id: Int): Int = {
    val clockChanged = if (isUniqueMin(id)) {
      minClock += 1
      logDebug(s"tick clock $id to ${id2clock(id) + 1}, and change min clock to $minClock")
      minClock
    } else {
      logDebug(s"tick clock $id to ${id2clock(id) + 1}, and don't change min clock: $minClock")
      0
    }
    id2clock(id) = id2clock(id) + 1
    clockChanged
  }

  def tickUntil(id: Int, clock: Int): Int = {
    val currentClock = getClock(id)
    val numTicks = clock - currentClock
    var newClock = 0
    for (i <- 0 until numTicks) {
      val clockChanged = tick(id)
      if (clockChanged != 0) {
        newClock = clockChanged
      }
    }
    newClock
  }

  def apply(id: Int): Int = {
    getClock(id)
  }

  def getClock(id: Int): Int = {
    require(id2clock.contains(id), "ps.common.VectorClock doesn't contain key: " + id)
    id2clock(id)
  }

  def getMinClock(): Int = {
    logDebug(s"current min clock is $minClock")
    minClock
  }

  def removeClock(id: Int): Int = {
    if (id2clock.size == 1) {
      id2clock.remove(id)
      minClock = -1
      minClock
    } else if (isUniqueMin(id)) {
      id2clock.remove(id)
      minClock = id2clock.values.min
      minClock
    } else {
      id2clock.remove(id)
      0
    }
  }

  private def isUniqueMin(id: Int): Boolean = {
    if (id2clock(id) != minClock) {
      return false
    }

    id2clock.count(_._2 == minClock) == 1
  }
}
