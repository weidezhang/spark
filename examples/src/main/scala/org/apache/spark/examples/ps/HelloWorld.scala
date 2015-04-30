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

package org.apache.spark.examples.ps

import org.apache.spark.ps.PSContext
import org.apache.spark.ps.local.LocalPSClient
import org.apache.spark.{SparkConf, SparkContext}


object HelloWorld {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Hello World to Test Parameter Server")
    val sc = new SparkContext(conf)

    val rdd = sc.parallelize(Array(1, 2, 3, 4, 5), 5)


    val psContext = new PSContext(sc)
    psContext.start()
    val masterUrl = psContext.masterUrl

    rdd.mapPartitionsWithIndex { (indexId, iter) =>
      val arr = iter.toArray
      val client = new LocalPSClient(indexId, masterUrl)

      for (i <- 0 to 10) {
        val a = client.get(1)
        val b
      }

      arr.iterator
    }

  }
}
