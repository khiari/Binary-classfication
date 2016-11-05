import sbt._

object Version {
  val hadoop    = "2.6.0"
  val logback   = "1.1.2"
  val mockito   = "1.10.19"
  val scala     = "2.11.7"
  val scalaTest = "2.2.4"
  val slf4j     = "1.7.6"
  val spark     = "1.6.2"
}

object Library {
  val hadoopClient   = "org.apache.hadoop" %  "hadoop-client"   % Version.hadoop
  val logbackClassic = "ch.qos.logback"    %  "logback-classic" % Version.logback
  val mockitoAll     = "org.mockito"       %  "mockito-all"     % Version.mockito
  val scalaTest      = "org.scalatest"     %% "scalatest"       % Version.scalaTest
  val slf4jApi       = "org.slf4j"         %  "slf4j-api"       % Version.slf4j
  val sparkSQL       = "org.apache.spark"  %% "spark-sql" % Version.spark
  val sparkMLlib     = "org.apache.spark" %% "spark-mllib" % Version.spark
  val sparkCore     = "org.apache.spark" %% "spark-core" % Version.spark
  val hadoopCommon ="org.apache.hadoop" % "hadoop-common" % "2.6.0" exclude ("org.apache.hadoop","hadoop-yarn-server-web-proxy")
 // val sparkMl     = "org.apache.spark" %% "spark-ml" % Version.spark
  val mongoSparkConnector = "org.mongodb.spark" % "mongo-spark-connector_2.11" % "0.1"

}

object Dependencies {
  import Library._

  val sparkAkkaHadoop = Seq(
    sparkSQL,
    sparkMLlib,
  //  sparkMl,
    sparkCore,
 // hadoopCommon,
    mongoSparkConnector,
  // hadoopClient,
    logbackClassic % "test",
    scalaTest      % "test",
    mockitoAll     % "test"
  )
}
