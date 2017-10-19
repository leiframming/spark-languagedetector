package org.apache.spark.ml.feature.languagedetection

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.languagedetection.LanguageDetectorModel
import org.specs2.mutable.Specification

class LanguageDetectorModelSpecs extends Specification {

  val logger = Logger.getLogger("org")
  logger.setLevel(Level.ERROR)

  import Spark.session.implicits._
  "The detector can" >> {

    "predict the language probability of a text" >> {
      val supportedLanguages = Array("de", "en")
      val gramLengths = Seq(3)
      val predictionData = Seq(
        "Dies ist ein deutscher Text, das ist ja sehr schön",
        "Dies ist ein andere deutscher Text, und der ist auch sehr schön",
        "This is a text in english, and that is very nice",
        "This is another text in english and that is also nice"
      )
        .toDF("fulltext")

      val probabilites = Map(
        ("Die".getBytes("UTF-8").toSeq, Array(1.0, 0.0)),
        ("Thi".getBytes("UTF-8").toSeq, Array(0.0, 1.0))
      )

      val model = new LanguageDetectorModel(
        gramProbabilities = probabilites,
        gramLengths = Seq(3),
        languages = Array("de", "en")
      )

      val transformed = model.transform(predictionData).as[(String, String)].collect

      transformed.count(_._2 == "de") must be_==(2)
      transformed.count(_._2 == "en") must be_==(2)

      transformed.size must be_==(4)

    }


  }



}
