package ml


import com.mongodb.spark.config.ReadConfig
import demo.SparkCommons
import com.mongodb.spark._

/**
 * Created by abderrahmen on 29/10/2016.
 */
object Classifier {

// loading data from mongodb database
 lazy val readConfig = ReadConfig(SparkCommons.sc)
 lazy  val train_rdd = SparkCommons.sc.loadFromMongoDB(readConfig=readConfig)







}
