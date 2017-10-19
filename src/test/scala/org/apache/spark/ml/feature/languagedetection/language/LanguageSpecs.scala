package org.apache.spark.ml.feature.languagedetection.language

import org.apache.spark.ml.feature.languagedetection.language.Language
import org.specs2.mutable.Specification


class LanguageSpecs extends Specification {


  "We can get a language" >> {
    val lang = Language.withName("de")
    lang.toString must be_==("de")
  }

}
