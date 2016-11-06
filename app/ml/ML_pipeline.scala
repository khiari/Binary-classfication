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
 * @author Abderrahmen khiari
 * @note this class encapsulate all the algorithm the algorithms for working with features
 *       (convert,index,encode,vectorize) and methods to fit and evaluate models
 */
trait ML_pipeline {

// file path to save model
  val LrModelFileName="conf\\ML_models\\spark-LR-model"
  val DtreeModelFileName="conf\\ML_models\\spark-DT-model"
  val RandomForestModelFileName="conf\\ML_models\\spark-RF-model"
  val NaiveBayesModelFileName="conf\\ML_models\\spark-NB-model"
  val GbtModelFileName="conf\\ML_models\\spark-GBT-model"
  val NeuralNetModelFileName="conf\\ML_models\\spark-NNet-model"


  var prediction_model :PipelineModel =null


  // load train data from mongodb using mongo-spark connector
  var readConfig = ReadConfig("test","train",Some("mongodb://host:port/"))
  var train_df = SparkCommons.sc.loadFromMongoDB(readConfig=readConfig).toDF()
  // load test data from mongodb using mongo-spark connector
  readConfig = ReadConfig("test","test",Some("mongodb://host:port/"))
  var test_df = SparkCommons.sc.loadFromMongoDB(readConfig=readConfig).toDF()


  train_df=train_df.withColumnRenamed("class","label")
  test_df=test_df.withColumnRenamed("class","label")

  val labelIndexer = new StringIndexer().setHandleInvalid("skip").setInputCol("label").setOutputCol("labelIndex")
  val indexerModel= labelIndexer.fit(train_df)
  train_df=indexerModel.transform(train_df).drop("label").withColumnRenamed("labelIndex","label")
  test_df= indexerModel.transform(test_df).drop("label").withColumnRenamed("labelIndex","label")


  // indexers used to index categorical features
  val workclassIndexer = new StringIndexer().setInputCol("workclass").setOutputCol("workclassIndex")
  val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
  val maritalStatusIndexer = new StringIndexer().setInputCol("marital-status").setOutputCol("marital-statusIndex")
  val occupationIndexer = new StringIndexer().setInputCol("occupation").setOutputCol("occupationIndex")
  val relationshipIndexer = new StringIndexer().setInputCol("relationship").setOutputCol("relationshipIndex")
  val raceIndexer = new StringIndexer().setInputCol("race").setOutputCol("raceIndex")
  val sexIndexer = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex")
  val nativeCountryIndexer = new StringIndexer().setInputCol("native-country").setOutputCol("native-countryIndex")

  //encoders to encode indexed features
  val workclassEncoder = new OneHotEncoder().setInputCol("workclassIndex").setOutputCol("workclassVec")
  val educationEncoder = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("educationVec")
  val maritalStatusEncoder = new OneHotEncoder().setInputCol("marital-statusIndex").setOutputCol("marital-statusVec")
  val occupationEncoder = new OneHotEncoder().setInputCol("occupationIndex").setOutputCol("occupationVec")
  val relationshipEncoder = new OneHotEncoder().setInputCol("relationshipIndex").setOutputCol("relationshipVec")
  val raceEncoder = new OneHotEncoder().setInputCol("raceIndex").setOutputCol("raceVec")
  val sexEncoder = new OneHotEncoder().setInputCol("sexIndex").setOutputCol("sexVec")
  val nativeCountryEncoder = new OneHotEncoder().setInputCol("native-countryIndex").setOutputCol("native-countryVec")

  /** =fitModel=
   * @note fit the model with provided pipeline and save it to hard disk
   *
   *
   * @param modelFileName the file path where to save the model after being generated
   */
  def fitModel(modelFileName:String) {

    println(train_df.printSchema())
    prediction_model = pipeline.fit(train_df)
    val holdout = prediction_model.transform(test_df).select("prediction","label")
    println(holdout.show(10))
    println(holdout.printSchema())

    prediction_model.write.overwrite().save(modelFileName)

  }

  /**
   * @note transform test data and evaluation the model using precision, recall, Fscore and ROC
   * @param test_data data to evaluate the model with
   */

  def evaluate_model(test_data:DataFrame): Unit ={


    val prediction_label = prediction_model.transform(test_data).select("prediction","label")

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
