package ml


import com.mongodb.spark.config.ReadConfig
import controllers.SparkCommons
import com.mongodb.spark._
import org.apache.spark.ml.{Pipeline, Transformer,PipelineModel}
import org.apache.spark.sql.{DataFrame}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}




/**
 * Created by abderrahmen on 29/10/2016.
 */
object LogisticRegression {



  // parameters to pass to LR algo
  case class Params(

                     regParam: Double = 0.0,
                     elasticNetParam: Double = 0.0,
                     maxIter: Int = 100,
                     fitIntercept: Boolean = true,
                     tol: Double = 1E-6)




    // var to store the score
    var f1_score = 0.0
  var LR = new LogisticRegression()

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

  val assembler = new VectorAssembler()
    .setInputCols(Array("workclassIndex", "educationIndex", "marital-statusIndex", "occupationIndex",
      "relationshipIndex","raceIndex","sexIndex","native-countryIndex","age","fnlwgt","education-num",
      "capital-gain","capital-loss","hours-per-week"))
    .setOutputCol("features")


  def load_transform_split_df(){
    readConfig = ReadConfig("test","train",Some("mongodb://host:port/"))
    train_df = SparkCommons.sc.loadFromMongoDB(readConfig=readConfig).toDF()
    train_df= df_transformer(train_df)
     val splits=train_df.randomSplit(Array(0.8,0.2),11L)
    train_df=splits(0)
    test_df=splits(1)

  }

/* transform the categorial columns to numerical columns
    the output dataframe has the folowing schema

  |-- features: vector (nullable = true)
  |-- label: double (nullable = true)
  */
def df_transformer(df  :DataFrame): DataFrame={
  var df_new=df
  println(df_new.show(1))
  val indexer = new StringIndexer()
  var cols =Array("workclass","education","marital-status","occupation","relationship","race","sex","native-country")
  for(col <- cols){
  df_new =indexer.setHandleInvalid("skip").setInputCol(col).setOutputCol(col+"index").fit(df_new).transform(df_new).drop(col).withColumnRenamed(col+"index",col)

  }


  val assembler = new VectorAssembler()
    .setInputCols(Array("age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race", "sex","capital-gain","capital-loss","hours-per-week","native-country"))
    .setOutputCol("features")
  df_new = assembler.transform(df_new)

  cols =Array("age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race", "sex","capital-gain","capital-loss","hours-per-week","native-country")


  for(col <- cols){

    df_new=df_new.drop(col)
  }


  df_new=df_new.drop("_id")

  // difine an udf to transform the class column
  def myFunc: (String => Double) =
  {
    s => if (s==">50K" || s==">50K.") 1.0 else 0.0
  }

  val myUDF = udf(myFunc)


   df_new = df_new.withColumn("class_x", myUDF(df_new("class"))).drop("class").withColumnRenamed("class_x","label")

df_new
}

  // parameterize  LR and fit the train data
def fit_df(params:Params) :PipelineModel = {

 LR.setFeaturesCol("features")
   .setLabelCol("label")
   .setRegParam(params.regParam)
  .setElasticNetParam(params.elasticNetParam)
  .setFitIntercept(params.fitIntercept)
  .setMaxIter(params.maxIter)
  .setTol(params.tol)

    val LR_pipeline = new Pipeline()
      .setStages(Array( workclassIndexer, educationIndexer, maritalStatusIndexer, occupationIndexer, relationshipIndexer, raceIndexer, sexIndexer, nativeCountryIndexer,
        workclassEncoder, educationEncoder, maritalStatusEncoder, occupationEncoder, relationshipEncoder, raceEncoder, sexEncoder, nativeCountryEncoder,
        assembler,LR))

 var model= LR_pipeline.fit(train_df)
    //System.setProperty("hadoop.home.dir", "C:\\hadoop-winutils-2.6.0\\")
    model.write.overwrite().save("spark-LR-model")
   var pipelineModel = PipelineModel.load("spark-LR-model")

    val holdout = pipelineModel.transform(test_df).select("prediction","label")
    println(holdout.show(10))
    println(holdout.printSchema())


  pipelineModel
}

  // method to evaluate the model
  def evaluateClassificationModel(
                                   model: PipelineModel,
                                   data: DataFrame,
                                   labelColName: String): Unit = {

    // run the model on the test data
    val fullPredictions = model.transform(data).cache()
    // get the prediction column
    val predictions = fullPredictions.select("prediction").rdd.map(_.getDouble(0))
    // get the label column
    val labels = fullPredictions.select(labelColName).rdd.map(_.getDouble(0))


    val multiclassMetrics = new MulticlassMetrics(predictions.zip(labels))
    // getting the F1_score
    f1_score = multiclassMetrics.fMeasure(0,1)

    println(s"  F1_score: $f1_score")
  }


  // encapsulate all the ML pipeline (loading,transforming,spliting,modeling,scoring)
  def run(params:Params){

    //load_transform_split_df()

    //println(train_df.printSchema())
    //println(test_df.printSchema())

    val LR_model=  LogisticRegression.fit_df(params)



   // evaluateClassificationModel(LR_model,test_df,"label")




  }


}
