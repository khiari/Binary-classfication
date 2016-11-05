package ml

import com.mongodb.spark.config.ReadConfig
import controllers.SparkCommons
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorIndexer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplitModel, TrainValidationSplit}
import org.apache.spark.ml.classification.{NaiveBayes, RandomForestClassifier, DecisionTreeClassifier, DecisionTreeClassificationModel}
import com.mongodb.spark._
import org.apache.spark.ml.{Pipeline}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

/**
 * Created by abderrahmen on 04/11/2016.
 */
object Dtree_pipeline {


  var prediction_model :TrainValidationSplitModel =null
  // load train  data from mongodb database
  var readConfig = ReadConfig("test","train",Some("mongodb://localhost:port/"))
  //var readConfig = ReadConfig("khiaridb","train",Some("mongodb://khiari:Kh_20843265@ds161475.mlab.com:61475/"))
  var train_df = SparkCommons.sc.loadFromMongoDB(readConfig=readConfig).toDF()
  readConfig = ReadConfig("test","test",Some("mongodb://localhost:port/"))
  //readConfig = ReadConfig("khiaridb","test",Some("mongodb://khiari:Kh_20843265@ds161475.mlab.com:61475/"))
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


  val assembler = new VectorAssembler()
    .setInputCols(Array("workclassIndex", "educationIndex", "marital-statusIndex", "occupationIndex",
      "relationshipIndex","raceIndex","sexIndex","native-countryIndex","age","fnlwgt","education-num",
      "capital-gain","capital-loss","hours-per-week"))
    .setOutputCol("features_vect")

  val vectorIndexer = new VectorIndexer().setInputCol("features_vect").setOutputCol("features").setMaxCategories(42)


  def dtree_pipeline():TrainValidationSplit={

    val dtree = new DecisionTreeClassifier()
   /*
   val paramGrid = new ParamGridBuilder()
      .addGrid(dtree.maxDepth,Array(5,7,10,15,20,18,25
       ))
      .addGrid(dtree.maxBins,Array(80))
      .addGrid(dtree.minInstancesPerNode,Array(200//,100,20,50
       ))
      .build()
*/
   val randomForest = new RandomForestClassifier()
 val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.maxDepth,Array(15,20,25
      ))
      .addGrid(randomForest.numTrees,Array(30,40,50,60,70,100))
      .addGrid(randomForest.maxBins,Array(80))
      .addGrid(randomForest.minInstancesPerNode,Array(200//,100,20,50
      ))
      .build()


    val naiveBayes = new NaiveBayes()
   /* val paramGrid = new ParamGridBuilder()

      .build()*/

    val pipeline = new Pipeline()
      .setStages(Array(workclassIndexer, educationIndexer, maritalStatusIndexer, occupationIndexer,
        relationshipIndexer, raceIndexer, sexIndexer, nativeCountryIndexer,assembler,vectorIndexer,randomForest))

    val tvs = new TrainValidationSplit()
    .setEstimator(pipeline)
    .setEvaluator(new BinaryClassificationEvaluator())
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.8)

  tvs
  }

  def fitModel(tvs:TrainValidationSplit) {


    println(train_df.printSchema())
    //training.withColumnRenamed("classIndex","label")
    prediction_model = tvs.fit(train_df)
    println(test_df.printSchema())
    val holdout = prediction_model.transform(test_df).select("prediction","label")
    println(holdout.show(10))
    println(holdout.printSchema())

    // have to do a type conversion for RegressionMetrics
    val metrics = new BinaryClassificationMetrics(holdout.rdd.map(x =>
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
