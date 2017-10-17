package org.apache.spark.ml.feature.languagedetection

import java.util.UUID

import breeze.linalg.argmax
import org.apache.spark.ml.{Model, Transformer}
import org.apache.spark.ml.feature.languagedetection.language.Language
import org.apache.spark.ml.linalg.{BLAS, DenseVector}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.sql.catalyst.encoders.{ExpressionEncoder, RowEncoder}
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.util.Random


object LanguageDetectorModel {



  /**
    * Ugly but fast pairwise log odds as difference between the odds in log space
    * @param vec
    * @return
    */
  def computePairwiseLogOdds(vec: DenseVector): Array[Double] = {
    val arr = vec.toArray
    val l = arr.length
    val resultVector = Array.fill(l * l)(0.0)
    arr.indices.dropRight(1).foreach{
      i => arr.indices.drop(i+1).foreach{
        j => resultVector(i*l + j) = arr(i) - arr(j)
      }
    }
    resultVector
  }

  /**
    * Detect the language: Compute grams from the byte array and compute the
    * language probabilities.
    * @param text
    */
  def detect(text: Array[Byte],
             probabilityMap: Map[Seq[Byte], Array[Double]],
             supportedLanguages: Seq[String],
             gramLengths: Seq[Int]): String = {
    val random = new Random()
    val defaultProb = Array.fill(supportedLanguages.length)(0.0).map(_ => random.nextGaussian())
    val probabilities = new DenseVector(Array.fill(supportedLanguages.length)(0.0))

    gramLengths
      .foreach { gramLength =>
        text
          .toIterator
          .sliding(gramLength)
          .foreach { byteSeq =>
            probabilityMap
              .get(byteSeq)
              .foreach{ probs =>
                val gramProb = new DenseVector(probs)
                BLAS.axpy(1.0, gramProb, probabilities)
              }
          }
      }

    val maxProbIndex = argmax(probabilities.toArray)
    supportedLanguages(maxProbIndex)
  }

  def detect(text: String, probabilityMap: Map[Seq[Byte], Array[Double]],
             supportedLanguages: Seq[String],
             gramLengths: Seq[Int]): String = {
    detect(text.toCharArray.map(_.toByte),
      probabilityMap,
      supportedLanguages,
      gramLengths)
  }

}

/**
  * Detect languages based on their n-gram probabilities
  *
  * gramProbabilities is a map containing the log-odds of a gram (in byte form)
  * belonging to a language. The Array of probabilities must be in the same order
  * as the sequence of languages. GramLenghts defines the length of the grams
  * for which the probabilities-map is defined.
  *
  */
class LanguageDetectorModel(override val uid: String,
                            val gramProbabilities: Map[Seq[Byte], Array[Double]],
                            val gramLenghts: Seq[Int],
                            val supportedLanguages: Seq[String])
  extends Model[LanguageDetectorModel]
    with HasInputCol
    with HasOutputCol {

  def this(gramProbabilities:  Map[Seq[Byte], Array[Double]],
           gramLengths: Seq[Int],
           languages: Seq[String]) = {
    this(
      uid = Identifiable.randomUID("LanguageDetectorModel"),
      gramProbabilities,
      gramLengths,
      languages
    )
  }

  setDefault(
    inputCol -> "fulltext",
    outputCol -> "lang"
  )

  override def transformSchema(schema: StructType): StructType =  {
    val inputType = schema(getInputCol).dataType
    require(inputType.sameType(StringType), s"Input type must be StringType but got $inputType.")
    SchemaUtils.appendColumn(schema, getOutputCol, StringType, nullable = true)
  }

  override def copy(extra: ParamMap) = defaultCopy(extra)

  /**
    * Detect the language of the text
    * @param dataset
    * @return
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._
    val schema = transformSchema(dataset.schema)
    val bProbabilitiesMap = dataset.sparkSession.sparkContext.broadcast(gramProbabilities)

    implicit val encoder: ExpressionEncoder[Row] = RowEncoder(schema)
    dataset
        .toDF()
        .map{row =>

          val textColInd = row.fieldIndex(getInputCol)
          val text = row.getString(textColInd)

          val detectedLanguage = LanguageDetectorModel.detect(
            text,
            bProbabilitiesMap.value,
            supportedLanguages,
            gramLenghts)
          Row.fromSeq(row.toSeq :+ detectedLanguage.toString)
      }(encoder)
      .toDF
  }

}