package org.apache.spark.ml.feature.languagedetection

import java.nio.charset.Charset
import java.util
import java.util.Locale

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SaveMode}
import org.apache.spark.sql.types.StructType


object LanguageDetector extends Logging {

  /**
    * Compute grams for each text
    * @param data
    * @param gramLengths
    * @return
    */
  private[this] def computeGrams(data: Dataset[(String, String)],
                   gramLengths: Seq[Int]): Dataset[(String, Seq[Byte], Int)] = {
    import data.sparkSession.implicits._

    data
      .flatMap{
        case (lang, text) =>
          gramLengths
            .flatMap{
              gramLength =>
                // Compute the occurrences of gram per language
                text
                  .getBytes(Charset.forName("UTF-8"))
                  .toSeq
                  .sliding(gramLength)
                  .toSeq
                  .groupBy(identity)
                  .mapValues(_.size)
                  .map{case (gram, count) => (lang, gram, count)}
            }
      }
  }

  /**
    * For each gram: Sum up the counts
    * @param grams
    */
  private[this] def reduceGrams( grams: Dataset[(String, Seq[Byte], Int)],
                   supportedLanguages: Seq[String]
                 ): Dataset[(String, Seq[Byte], Int)] = {
    import grams.sparkSession.implicits._

    supportedLanguages
      .map{
        lang => grams
          .filter(a => a._1 == lang.toString)
          .groupByKey{case (language, gram, count) => gram}
          .reduceGroups((a,b) => (a._1, a._2, a._3 + b._3))
          .map{case (key, rest) => rest}
      }
      .reduce(_ union _)
  }


  /**
    * Compute the probability of occurrence of a gram in each language
    * We use the odds as probability: #Occurrence in language / #Overall
    * and compute Log(1.0 + P)
    */

  private[this] def computeProbabilities(grams: Dataset[(String, Seq[Byte], Int)],
                           supportedLanguages: Seq[String]): Dataset[(Seq[Byte], Array[Double])] = {
    import grams.sparkSession.implicits._

    grams
      .groupByKey{case (lang, gram, count) => gram}
      .mapGroups{
        case (gram, it) =>
          val itSeq = it.toSeq

          val langProbs = supportedLanguages
            .map{lang => itSeq.count(_._1 == lang).toDouble / itSeq.size.toDouble}
            .map(d => Math.log(1.0 + d))
            .toArray

          (gram, langProbs)
      }
  }


  /**
    * For each language l take the top k grams that have a high value of P(l) for each
    *
    * @param gramProbabilities
    */
  private[this] def filterTopGrams(
                      gramProbabilities: Dataset[(Seq[Byte], Array[Double])],
                      supportedLanguages: Seq[String],
                      languageProfileSize: Int
                    ) = {
    import gramProbabilities.sparkSession.implicits._

    val topGramSet = supportedLanguages
      .indices
      .map(i =>
        gramProbabilities
          .map{ case (gram, probs) => (supportedLanguages(i), gram, probs(i))}
          .groupByKey{case (lang, gram, prob) => lang}
          .flatMapGroups{
              case (lang, it) =>
                it
                  .toSeq
                  .sortBy(_._3)(Ordering.Double.reverse)
                  .take(languageProfileSize)
                  .map{case (l,g,p) => g}
            }
      )
      .reduce(_ union _)


    gramProbabilities
      .joinWith(
        topGramSet,
        gramProbabilities("_1") === topGramSet("value")
      )
      .map(_._1)

  }


  /**
    * Compute the probabilitie of a n-gram occurring in a particular language.
    * The gram is modeled as an array of bytes, the probability vector is an array of doubles
    * @param data Training data, Dataset[(Language, Fulltext)], i.e. Wikipedia dump
    * @param gramLengths Seq[Int] of the gram sizes that should be used
    * @param languageProfileSize Int, number of top-k grams that should be used
    * @param supportedLanguages Array[Language] of languages that can be detected. This is
    *                           also the order of the probability vector that is computed
    * @return
    */
  def computeGramProbabilities(
                                data: Dataset[(String, String)],
                                gramLengths: Seq[Int],
                                languageProfileSize: Int,
                                supportedLanguages: Seq[String]
                             ): Dataset[(Seq[Byte], Array[Double])] = {
    // Compute all grams from
    val grams = computeGrams(data, gramLengths)

    // Merge the counts for the grams of
    val reducedGrams = reduceGrams(grams, supportedLanguages)

    // For each gram: Compute the probability of occurrence for each language
    val probabilities = computeProbabilities(reducedGrams, supportedLanguages).cache()

    // Filter the n-grams for their language probability: Take only the top k values
    val topGrams = filterTopGrams(probabilities, supportedLanguages, languageProfileSize)

    probabilities.unpersist()
    topGrams
  }

  def save[T](saveFile: String, ds: Dataset[T]): Unit = {
    logInfo(s"Saving dataset to $saveFile")
    val writer = ds.write.format("parquet")
    writer.mode(SaveMode.Overwrite).save(saveFile)
  }

}


class LanguageDetector(
                        val uid: String,
                        val supportedLanguages: Seq[String],
                        val gramLengths: Seq[Int],
                        val languageProfileSize: Int
                      )
  extends Estimator[LanguageDetectorModel]
    with HasInputCol with HasLabelCol {

  def this(
            supportedLanguages: Seq[String],
            gramLengths: Seq[Int],
            languageProfileSize: Int) = this(
    Identifiable.randomUID("LanguageDetector"),
    supportedLanguages,
    gramLengths,
    languageProfileSize
  )

  setDefault(
    inputCol -> "fulltext",
    labelCol -> "lang"
  )

  def setInputCol(value: String): LanguageDetector.this.type = set(inputCol, value)
  def setLabelCol(value: String): LanguageDetector.this.type = set(labelCol, value)

  val saveGramsToHDFS = new Param[Option[String]](this, "saveGrams", "Persist the dataset of grams to HDFS")
  def setSaveGramsToHDFS(value: Option[String]): LanguageDetector.this.type = set(saveGramsToHDFS, value)
  setDefault(saveGramsToHDFS -> None)

  override def transformSchema(schema: StructType): StructType = schema
  override def copy(extra: ParamMap): Estimator[LanguageDetectorModel] = defaultCopy(extra)

  override def fit(dataset: Dataset[_]): LanguageDetectorModel = {
    import dataset.sparkSession.implicits._

    // InputTrainingData is a dataset [String, Int, String] which is [LanguageName, Id, Fulltext]
    val inputTrainingData = dataset
      .select($(labelCol), $(inputCol))
      .as[(String, String)]
      .cache()


    // Check if all input training data are from supported languages
    inputTrainingData
      .map(_._1)
      .distinct
      .foreach{lang =>
        if (!supportedLanguages.contains(lang))
          throw new Exception(s"Input data contians $lang, but it is not " +
            s"in the list of supported languages")
      }


    // Check if input training data contains values for each language
    supportedLanguages.foreach(
      lang =>
        inputTrainingData.filter(langTextPair => langTextPair._1 == lang).count match {
          case 0 => throw new Exception(s"No training examples found for language $lang. Provide examples for each language")
          case _ =>
        }
    )

    val gramProbabilities: Dataset[(Seq[Byte], Array[Double])] = LanguageDetector.computeGramProbabilities(
      inputTrainingData,
      gramLengths,
      languageProfileSize,
      supportedLanguages
    ).cache()

    inputTrainingData.unpersist()


    $(saveGramsToHDFS).foreach(s =>LanguageDetector.save(s, gramProbabilities))


    val probabilitiesMap: Map[Seq[Byte], Array[Double]] = gramProbabilities
      .collect
      .toMap


    gramProbabilities.unpersist()

    new LanguageDetectorModel(
      gramProbabilities = probabilitiesMap,
      gramLengths = gramLengths,
      languages = supportedLanguages
    )
  }
}
