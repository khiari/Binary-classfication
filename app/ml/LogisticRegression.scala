package ml


import com.mongodb.spark.config.ReadConfig
import demo.SparkCommons
import com.mongodb.spark._
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.{DataFrame}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.{MulticlassMetrics}



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
  // load train  data from mongodb database
   var readConfig = ReadConfig("test","train",Some("mongodb://host:port/"))
   var train_df = SparkCommons.sc.loadFromMongoDB(readConfig=readConfig).toDF()
 // load test data from mongodb database
  readConfig = ReadConfig("test","test",Some("mongodb://host:port/"))
  var test_df= SparkCommons.sc.loadFromMongoDB(readConfig=readConfig).toDF()

  var LR = new LogisticRegression()





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
def fit_df(params:Params) : LogisticRegressionModel = {

 LR.setFeaturesCol("features")
   .setLabelCol("label")
   .setRegParam(params.regParam)
  .setElasticNetParam(params.elasticNetParam)
  .setFitIntercept(params.fitIntercept)
  .setMaxIter(params.maxIter)
  .setTol(params.tol)

 val model=LR.fit(train_df)

  model
}

  // method to evaluate the model
  def evaluateClassificationModel(
                                   model: Transformer,
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

    load_transform_split_df()

    println(train_df.printSchema())
    println(test_df.printSchema())

    val LR_model=  LogisticRegression.fit_df(params)

    println(LR_model.coefficients)

    evaluateClassificationModel(LR_model,test_df,"label")




  }





}
