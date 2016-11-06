package models

import controllers.Assets
import play.api.libs.json.Json


case class Person(age:Int,workclass:String,fnlwgt:Int,education:String,educationNum:Int ,maritalStatus:String

,occupation:String,relationship:String,race:String,sex:String
                  ,capitalGain:Int,capitalLoss:Int
                  ,hoursPerWeek:Int,nativeCountry:String,mlModel:String
)


  object  Person{

    implicit  val personFormat = Json.format[Person]
  }


