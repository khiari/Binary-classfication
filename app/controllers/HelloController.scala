package controllers

import ml.LogisticRegression.Params
import ml._
import models.Person
import play.api.Play
import play.api.data.Forms._
import play.api.data.Form
import play.api.libs.json.Json
import play.api.mvc._


object HelloController extends Controller {

  val personForm:Form[Person]=Form{mapping("age"->number,"workclass"->text,"fnlwgt"->number,"education"->text,"educationNum"->number,"maritalStatus"->text

    , "occupation"->text,"relationship"->text,"race"->text,"sex"->text
    ,"capitalGain"->number, "capitalLoss"->number,"hoursPerWeek"->number,"nativeCountry"->text
  )(Person.apply)(Person.unapply)}

  var result = new String()


  def index() = Action {
    Ok(views.html.index(""))
  }



  def logisticRegression=Action{
    // this model scored  0.9000482858522453 using F1_scoring
   // LogisticRegression.run(Params(0.0,0.0,100,true,1E-6))
    //LogisticRegression.test_OHE()
    LR_pipeline.fitModel(LR_pipeline.preppedLRPipeline())
    Ok(s"logistic regression model scored:${LogisticRegression.f1_score} using F1 scoring")

  }



  def predictIncome= Action{
    implicit request => val person = personForm.bindFromRequest.get


    Ok(views.html.predictionResult(LR_pipeline.getResult(person)))

  }




}
