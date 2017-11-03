package org.apache.spark.ml.feature.languagedetection.preprocessing

import java.util.Locale

import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.catalyst.encoders.RowEncoder

/**
  * Simple transformer to remove excess whitespace and symbols like "<[]>/\
  *
  */
class SpecialCharPreprocessor(override val uid: String)
  extends Transformer
    with HasOutputCol
    with Logging {

  def this() = this(Identifiable.randomUID("SpecialCharPreprocessor"))

  setDefault(
    outputCol -> "fulltext"
  )

  def setInputCol(value: String): SpecialCharPreprocessor.this.type = set(outputCol, value)

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val filteredSchema = schema.filter(structField => structField.name != $(outputCol))
    val filteredStruct = StructType(filteredSchema)
    SchemaUtils.appendColumn(filteredStruct, $(outputCol), StringType, nullable = true)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._

    implicit val encoder = RowEncoder(transformSchema(dataset.schema))

    dataset
      .toDF()
      .map{
        (row: Row) =>

          val textColInd = row.fieldIndex($(outputCol))
          val text = row.getString(textColInd)

          // Convert text to lower case based on the locale of the data
          val replText = text
            .replaceAll("/_[]*()%^&@$#:|{}<>~`\"\\", "")
            .replaceAll("  *", "")

          // Put it back into the data frame
          val rowseq = row
            .toSeq
            .toArray
            .zipWithIndex
            .filter(ai => ai._2 != textColInd)
            .map(_._1)

          // Add the new column
          val r: Row = new GenericRowWithSchema(rowseq ++ Array(replText), transformSchema(row.schema))
          r
      }
  }
}
