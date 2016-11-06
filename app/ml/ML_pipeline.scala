package ml

import com.mongodb.spark.config.ReadConfig
import controllers.SparkCommons
import ml.LR_pipeline._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import com.mongodb.spark._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame

/**
 * Created by abderrahmen on 06/11/2016.
 */
class ML_pipeline {

  val LrModelFileName="conf\\ML_models\\spark-LR-model"
  val DtreeModelFileName="spark-DT-model"
  val RandomForestModelFileName="spark-RF-model"
  val NaiveBayesModelFileName="spark-NB-model"
  val GbtModelFileName="spark-GBT-model"
  val NeuralNetModelFileName="spark-NNet-model"


  var prediction_model :PipelineModel =null
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

  def fitModel(modelFileName:String) {

    println(train_df.printSchema())
    prediction_model = pipeline.fit(train_df)
    val holdout = prediction_model.transform(test_df).select("prediction","label")
    println(holdout.show(10))
    println(holdout.printSchema())

    prediction_model.write.overwrite().save(modelFileName)

  }

  def evaluate_model(data:DataFrame): Unit ={


    val prediction_label = prediction_model.transform(data).select("prediction","label")

    val metrics = new BinaryClassificationMetrics(prediction_label.rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))


    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)




  }





}
