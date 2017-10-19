package org.apache.spark.ml.feature.languagedetection

import java.net.URI

import com.typesafe.config.ConfigFactory
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.specs2.mutable.Specification

class LanguageDetectionModelItSpecs extends Specification {
  // Force creation of a sparkcontext for the spec
  val session = Spark.session
  println(session.version)

  "We can save and load a model" >> {

    val fs = FileSystem.get(new Configuration())
    val conf = ConfigFactory.load()
    val saveFolder = conf.getString("languageDetector.testDir")
    val saveFolderPath = new Path(new URI(saveFolder))

    // Clean the slate for the test
    if (fs.exists(saveFolderPath))
      fs.delete(saveFolderPath, true)

    // Create a dummy model
    val model = new LanguageDetectorModel(
      Map("a".getBytes.toSeq -> Array(1.0)),
      Seq(1),
      Seq("a")
    )

    // Save the model to disk
    model.write.save(saveFolder)

    // Check if the model is there
    fs.exists(saveFolderPath) must be_==(true)

    // Load the model back again
    val model2 = LanguageDetectorModel.load(saveFolder)

    model.gramLenghts.size must be_==(1)

    // At the end: Delete the created folder
    fs.delete(saveFolderPath, true)

  }


}
