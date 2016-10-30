package demo

import ml.LogisticRegression.Params
import ml._
import play.api.mvc._


object HelloController extends Controller {

  def index = Action {
    Ok("hello world")
  }


  def logisticRegression=Action{
    // this model scored  0.9000482858522453 using F1_scoring
    LogisticRegression.run(Params(0.0,0.0,100,true,1E-6))
    Ok(s"logistic regression model scored:${LogisticRegression.f1_score} using F1 scoring")

  }

}
