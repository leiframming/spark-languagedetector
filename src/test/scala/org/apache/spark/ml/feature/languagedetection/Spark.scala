package org.apache.spark.ml.feature.languagedetection

import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object Spark {


  lazy val conf: SparkConf = new SparkConf()
    .setAppName("langDetect")
    .setMaster("local[4]")
    .set("spark.cores.max", "4")


  lazy val session: SparkSession = SparkSession
    .builder()
    .config(conf)
    .getOrCreate()

  lazy val sc: SparkContext = session.sparkContext

}
