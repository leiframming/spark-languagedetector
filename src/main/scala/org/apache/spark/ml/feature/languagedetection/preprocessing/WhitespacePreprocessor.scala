package org.apache.spark.ml.feature.languagedetection.preprocessing

import java.util.Locale

import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol}
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{StringType, StructType}

/**
  * Simple transformer to remove excess whitespace and symbols like "<[]>/\
  *
  */
class WhitespacePreprocessor
  extends Transformer
    with HasInputCol
    with Logging {

  setDefault(
    inputCol -> "fulltext",
  )

  def setInputCol(value: String): WhitespacePreprocessor.this.type = set(inputCol, value)

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val filteredSchema = schema.filter(structField => structField.name != $(inputCol))
    val filteredStruct = StructType(filteredSchema)
    SchemaUtils.appendColumn(filteredStruct, $(inputCol), StringType, nullable = true)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._

    val lowerCaseText = dataset
      .toDF()
      .map{
        (row: Row) =>

          val textColInd = row.fieldIndex($(inputCol))
          val text = row.getString(textColInd)

          // Convert text to lower case based on the locale of the data
          text.replaceAll("/\_[]*()%^&@$#:|{}<>~`\"\\")

          // Put it back into the data frame
          val rowseq = row
            .toSeq
            .zipWithIndex
            .filter(ai => ai._2 != textColInd)
            .map(_._1)

          // Add the new column
          Row.fromSeq(rowseq ++ Seq(lcText))
      }

    dataset.toDF

  }
}
