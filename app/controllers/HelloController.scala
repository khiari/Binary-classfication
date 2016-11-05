package controllers



import ml._
import models.Person
import play.api.data.Forms._
import play.api.data.Form
import play.api.mvc._
import org.apache.spark.ml.PipelineModel



object HelloController extends Controller {

  //System.setProperty("hadoop.home.dir", "C:\\hadoop-common-2.2.0-bin-master")

 // var readConfig = ReadConfig("khiaridb","train",Some("mongodb://khiari:Kh_20843265@ds161475.mlab.com:61475/"))
  //var train_df = SparkCommons.sc.loadFromMongoDB(readConfig = readConfig).toDF()
//  println(train_df.printSchema())

  val sparkcontext = SparkCommons.sc

  val personForm:Form[Person]=Form{mapping("age"->number,"workclass"->text,"fnlwgt"->number,"education"->text,"educationNum"->number,"maritalStatus"->text

    , "occupation"->text,"relationship"->text,"race"->text,"sex"->text
    ,"capitalGain"->number, "capitalLoss"->number,"hoursPerWeek"->number,"nativeCountry"->text
  )(Person.apply)(Person.unapply)}

  var result = new String()


  def index() = Action {
   // var readConfig = ReadConfig("khiaridb","train",Some("mongodb://khiari:Kh_20843265@ds161475.mlab.com:61475/"))
    //var train_df = SparkCommons.sc.loadFromMongoDB(readConfig = readConfig).toDF()
    //println(train_df.printSchema())
    println("ok")

    Ok(views.html.index("hello world !!"))
  }



  def logisticRegression=Action{
    // this model scored  0.9000482858522453 using F1_scoring
   //LogisticRegression.run(Params(0.0,0.0,100,true,1E-6))
    val pipelineModel= PipelineModel.load("spark-LR-model")


    //LogisticRegression.test_OHE()
   // LR_pipeline.fitModel(LR_pipeline.preppedLRPipeline())
    Ok("ok")

  }

  def decisionTree=Action{

   // Dtree_pipeline.fitModel((Dtree_pipeline.dtree_pipeline()))
    Ok("ok")

  }



  def predictIncome= Action{
    implicit request => val person = personForm.bindFromRequest.get


    Ok(views.html.predictionResult(LR_pipeline.getResult(person)))

  }




}
