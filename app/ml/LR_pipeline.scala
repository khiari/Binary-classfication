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


object LR_pipeline  extends ML_pipeline {


 /* var prediction_model :PipelineModel =null
  // load train  data from mongodb database

  var readConfig = ReadConfig("test","train",Some("mongodb://host:port/"))
  var train_df = SparkCommons.sc.loadFromMongoDB(readConfig=readConfig).toDF()

  readConfig = ReadConfig("test","test",Some("mongodb://host:port/"))
  var test_df = SparkCommons.sc.loadFromMongoDB(readConfig=readConfig).toDF()

  train_df=train_df.withColumnRenamed("class","label")
  test_df=test_df.withColumnRenamed("class","label")



  val columns =Array("age","workclass","fnlwgt","education","education-num","marital-status","occupation",
    "relationship","race", "sex","capital-gain","capital-loss","hours-per-week","native-country","class")

      val labelIndexer = new StringIndexer().setHandleInvalid("skip").setInputCol("label").setOutputCol("labelIndex")

     val indexerModel= labelIndexer.fit(train_df)
    train_df=indexerModel.transform(train_df).drop("label").withColumnRenamed("labelIndex","label")
    test_df= indexerModel.transform(test_df).drop("label").withColumnRenamed("labelIndex","label")


      val workclassIndexer = new StringIndexer().setInputCol("workclass").setOutputCol("workclassIndex")
      val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
      val maritalStatusIndexer = new StringIndexer().setInputCol("marital-status").setOutputCol("marital-statusIndex")
      val occupationIndexer = new StringIndexer().setInputCol("occupation").setOutputCol("occupationIndex")
      val relationshipIndexer = new StringIndexer().setInputCol("relationship").setOutputCol("relationshipIndex")
      val raceIndexer = new StringIndexer().setInputCol("race").setOutputCol("raceIndex")
      val sexIndexer = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex")
      val nativeCountryIndexer = new StringIndexer().setInputCol("native-country").setOutputCol("native-countryIndex")


  val workclassEncoder = new OneHotEncoder().setInputCol("workclassIndex").setOutputCol("workclassVec")
  val educationEncoder = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("educationVec")
  val maritalStatusEncoder = new OneHotEncoder().setInputCol("marital-statusIndex").setOutputCol("marital-statusVec")
  val occupationEncoder = new OneHotEncoder().setInputCol("occupationIndex").setOutputCol("occupationVec")
  val relationshipEncoder = new OneHotEncoder().setInputCol("relationshipIndex").setOutputCol("relationshipVec")
  val raceEncoder = new OneHotEncoder().setInputCol("raceIndex").setOutputCol("raceVec")
  val sexEncoder = new OneHotEncoder().setInputCol("sexIndex").setOutputCol("sexVec")
  val nativeCountryEncoder = new OneHotEncoder().setInputCol("native-countryIndex").setOutputCol("native-countryVec")

*/
  val assembler = new VectorAssembler()
    .setInputCols(Array("workclassIndex", "educationIndex", "marital-statusIndex", "occupationIndex",
      "relationshipIndex","raceIndex","sexIndex","native-countryIndex","age","fnlwgt","education-num",
      "capital-gain","capital-loss","hours-per-week"))
    .setOutputCol("features")

  val lr = new LogisticRegression()
  .setFeaturesCol("features")
  .setLabelCol("label")
  .setFitIntercept(true)
  .setElasticNetParam(0.0)
  .setRegParam(0.0)
  .setTol(1E-6)

  val pipeline = new Pipeline()
    .setStages(Array( workclassIndexer, educationIndexer, maritalStatusIndexer, occupationIndexer, relationshipIndexer, raceIndexer, sexIndexer, nativeCountryIndexer,
      workclassEncoder, educationEncoder, maritalStatusEncoder, occupationEncoder, relationshipEncoder, raceEncoder, sexEncoder, nativeCountryEncoder,
      assembler,lr))

 /* def preppedLRPipeline():TrainValidationSplit = {
    val lr = new LogisticRegression()

    val LR_paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0
        //,0.25, 0.5, 0.75, 1.0
      ))
      .build()

    val pipeline = new ML_pipeline()
      .setStages(Array( workclassIndexer, educationIndexer, maritalStatusIndexer, occupationIndexer, relationshipIndexer, raceIndexer, sexIndexer, nativeCountryIndexer,
        workclassEncoder, educationEncoder, maritalStatusEncoder, occupationEncoder, relationshipEncoder, raceEncoder, sexEncoder, nativeCountryEncoder,
        assembler,lr))

     val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(LR_paramGrid)
      .setTrainRatio(0.8)


    tvs
  }
  */



}
