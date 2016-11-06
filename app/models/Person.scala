package models

import play.api.libs.json.Json

/**
 *
 * @param age
 * @param workclass
 * @param fnlwgt
 * @param education
 * @param educationNum
 * @param maritalStatus
 * @param occupation
 * @param relationship
 * @param race
 * @param sex
 * @param capitalGain
 * @param capitalLoss
 * @param hoursPerWeek
 * @param nativeCountry
 * @param mlModel
 */

case class Person(age:Int,workclass:String,fnlwgt:Int,education:String,educationNum:Int ,maritalStatus:String,
                  occupation:String,relationship:String,race:String,sex:String,capitalGain:Int,capitalLoss:Int
                  ,hoursPerWeek:Int,nativeCountry:String,mlModel:String)


  object  Person{

    implicit  val personFormat = Json.format[Person]
  }


