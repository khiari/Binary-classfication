package demo

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Handles configuration, context and so
 *
 * @author Alexandre Masselot.
 */
object SparkCommons {
  //build the SparkConf  object at once
  lazy val conf = {
    new SparkConf(false)
      .setMaster("local[*]")
      .setAppName("play demo")
      .set("spark.logConf", "true")
      // param to let the spar-mongo-connecter connect to the database and get the train data
      .set("spark.mongodb.input.uri","mongodb://localhost:27017/test.train")
  }

  lazy val sc = SparkContext.getOrCreate(conf)
  lazy val sqlContext = new SQLContext(sc)


}
