package ml

import com.mongodb.spark.config.ReadConfig
import controllers.SparkCommons
import models.Person
import org.apache.log4j.Logger
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, OneHotEncoder, StringIndexer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit,TrainValidationSplitModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.DataFrame
import com.mongodb.spark._
import org.apache.spark.ml.{Pipeline, PipelineModel}

/**
 * @note an ml pipeline for logistic regression
 *
 */

object LR_pipeline  extends ML_pipeline {

  /**
   * @note is a transformer that combines a given list of columns into a single vector column
   */

  val assembler = new VectorAssembler()
    .setInputCols(Array("workclassIndex", "educationIndex", "marital-statusIndex", "occupationIndex",
      "relationshipIndex","raceIndex","sexIndex","native-countryIndex","age","fnlwgt","education-num",
      "capital-gain","capital-loss","hours-per-week"))
    .setOutputCol("features")

  /**
   *@note logistic regression algorithm. Currently, this class only supports binary classification
   *
   */
  val lr = new LogisticRegression()
  .setFeaturesCol("features")
  .setLabelCol("label")
  .setFitIntercept(true)
  .setElasticNetParam(0.0)
  .setRegParam(0.0)
  .setTol(1E-6)

  /**
   * @note API provided by spark to make it possible for developpers to run a sequence of algorithms to process and learn from data
   */
  val pipeline = new Pipeline()
    .setStages(Array( workclassIndexer, educationIndexer, maritalStatusIndexer, occupationIndexer, relationshipIndexer, raceIndexer, sexIndexer, nativeCountryIndexer,
      workclassEncoder, educationEncoder, maritalStatusEncoder, occupationEncoder, relationshipEncoder, raceEncoder, sexEncoder, nativeCountryEncoder,
      assembler,lr))





}
