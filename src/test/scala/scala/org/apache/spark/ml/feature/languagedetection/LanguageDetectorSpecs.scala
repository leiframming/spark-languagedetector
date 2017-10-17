package scala.org.apache.spark.ml.feature.languagedetection

import org.apache.log4j.{Level, Logger}
import org.specs2.mutable.Specification

class LanguageDetectorSpecs extends Specification {


  val logger = Logger.getLogger("org")
  logger.setLevel(Level.ERROR)


  "We can fit a basic model" >> {

    val supportedLanguages = Array("de", "en")
    val gramLengths = Seq(3)
    val trainingData = Seq(
      ("de", "Dies ist ein deutscher Text, das ist ja sehr schön"),
      ("de", "Dies ist ein andere deutscher Text, und der ist auch sehr schön"),
      ("en", "This is a text in english, and that is very nice"),
      ("en", "This is another text in english and that is also nice")
    )
    .toDF("lang", "fulltext")



    val detector = new LanguageDetector(
      supportedLanguages = supportedLanguages,
      gramLengths = gramLengths,
      languageProfileSize = 5
    )

    val model = detector.fit(trainingData)
    println(model.gramProbabilities)
    model.gramProbabilities.size must be_==(10)
    model.gramProbabilities.head._2.length must be_==(2)

  }


  "Languages that are not supported are filtered out" >> {
    val supportedLanguages = Array("de", "en")
    val gramLengths = Seq(3)
    val trainingData = Seq(
      ("de", "Dies ist ein deutscher Text, das ist ja sehr schön"),
      ("de", "Dies ist ein andere deutscher Text, und der ist auch sehr schön"),
      ("es", "Habla espanol"),
      ("es", "Donde est la bibliotheka")
    )
      .toDF("lang", "fulltext")



    val detector = new LanguageDetector(
      supportedLanguages = supportedLanguages,
      gramLengths = gramLengths,
      languageProfileSize = 5
    )

    detector.fit(trainingData) must throwA(new Exception("No training examples found for language en. Provide examples for each language"))



  }

}
