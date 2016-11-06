package ml

import com.mongodb.spark.config.ReadConfig
import controllers.SparkCommons
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorIndexer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplitModel, TrainValidationSplit}
import org.apache.spark.ml.classification.{GBTClassifier,NaiveBayes, RandomForestClassifier, DecisionTreeClassifier,
MultilayerPerceptronClassifier}
import com.mongodb.spark._
import org.apache.spark.ml.{Pipeline}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics


/**
 *
 *@note this object is used to generate and save all the models expect Logistic regression because they share the same pipeline
 */
object Dtree_pipeline extends ML_pipeline{



  val assembler = new VectorAssembler()
    .setInputCols(Array("workclassIndex", "educationIndex", "marital-statusIndex", "occupationIndex",
      "relationshipIndex","raceIndex","sexIndex","native-countryIndex","age","fnlwgt","education-num",
      "capital-gain","capital-loss","hours-per-week"))
    .setOutputCol("features_vect")

  val vectorIndexer = new VectorIndexer().setInputCol("features_vect").setOutputCol("features").setMaxCategories(42)

  val dtree = new DecisionTreeClassifier()
  .setMaxDepth(30)
  .setFeaturesCol("features")
  .setLabelCol("label")

  val randomForest = new RandomForestClassifier()
  .setMaxDepth(30)
  .setNumTrees(50)
  .setFeaturesCol("features")
  .setLabelCol("label")

  val naiveBayes = new NaiveBayes()

  val gbt = new GBTClassifier()
    .setMaxDepth(30)
  .setFeaturesCol("features")
  .setLabelCol("label")

  val layers = Array[Int](4, 5, 4, 3)
  // create the trainer and set its parameters
  val neuralNet = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setBlockSize(128)
    .setSeed(1234L)
    .setMaxIter(100)



  val pipeline = new Pipeline()
    .setStages(Array(workclassIndexer, educationIndexer, maritalStatusIndexer, occupationIndexer,
      relationshipIndexer, raceIndexer, sexIndexer, nativeCountryIndexer,assembler,vectorIndexer, randomForest//,gbt,dtree,naiveBayes,randomForest,neuralNet
       ))


}
