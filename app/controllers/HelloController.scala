package controllers



import ml._
import models.Person
import play.api.data.Forms._
import play.api.data.Form
import play.api.mvc._
import org.apache.spark.ml.PipelineModel


object HelloController extends Controller {

// path to the saved models
  val LrModelFileName="conf/ML_models/spark-LR-model"
  val DtreeModelFileName="conf/ML_models/spark-DT-model"
  val RandomForestModelFileName="conf/ML_models/spark-RF-model"
  val NaiveBayesModelFileName="conf/ML_models/spark-NB-model"
  val GbtModelFileName="conf/ML_models/spark-GBT-model"
  val NeuralNetModelFileName="conf/ML_models/spark-NNet-model"


  System.setProperty("hadoop.home.dir", "C:\\hadoop-common-2.2.0-bin-master")


  val sparkcontext = SparkCommons.sc

  val personForm:Form[Person]=Form{mapping("age"->number,"workclass"->text,"fnlwgt"->number,"education"->text,"educationNum"->number,
    "maritalStatus"->text, "occupation"->text,"relationship"->text,"race"->text,"sex"->text,"capitalGain"->number, "capitalLoss"->number,
    "hoursPerWeek"->number,"nativeCountry"->text,"mlModel"-> text)(Person.apply)(Person.unapply)}

  var result = new String()


  def index() = Action {
    Ok(views.html.index(""))
  }



  def logisticRegression=Action{
    LR_pipeline.fitModel(LrModelFileName)
    Ok("ok")

  }

  def decisionTree=Action{

   Dtree_pipeline.fitModel(RandomForestModelFileName)
    Ok("ok")

  }


  /**
   * @note this method use the specified model in the form to predict the income of the person   *
   */
  def predictIncome= Action{
    implicit request => val person = personForm.bindFromRequest.get

      val input_df=SparkCommons.sqlContext.createDataFrame(Seq((person.age,person.workclass,person.fnlwgt,person.education,person.educationNum,
        person.maritalStatus,person.occupation, person.relationship,person.race,person.sex,person.capitalGain,person.capitalLoss,person.hoursPerWeek,person.nativeCountry)))
        .toDF("age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex",
          "capital-gain","capital-loss","hours-per-week","native-country")

      val predictionModel= person.mlModel match {

        case "LR" => PipelineModel.load(LrModelFileName)
        case "DT" => PipelineModel.load(DtreeModelFileName)
        case "RF" => PipelineModel.load(RandomForestModelFileName)
        case "GBT"=> PipelineModel.load(GbtModelFileName)
        case "NN" => PipelineModel.load(NeuralNetModelFileName)
        case "NB" => PipelineModel.load(NaiveBayesModelFileName)
      }


      val result = predictionModel.transform(input_df).select("prediction").first().get(0)
      var msg=""
      if (result == 0.0)
        msg= s"this person ${person.toString} earn <=50K"
      else msg = s"this person ${person.toString} earn >50K"
      println(s"result: $msg")


      Ok(views.html.predictionResult(msg))

  }




}
