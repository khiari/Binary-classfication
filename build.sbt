import play.sbt.PlayImport._
import play.sbt.routes.RoutesKeys._
import com.typesafe.sbt.packager.MappingsHelper._

name         := """spark-play"""
organization := "Abderrahmen khiari"
version      := "0.0.1"
scalaVersion := Version.scala



lazy val root = (project in file(".")).enablePlugins(PlayScala)

scalaSource in Compile <<= baseDirectory / "src/scala"

libraryDependencies ++= Dependencies.sparkAkkaHadoop

dependencyOverrides ++= Set(
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
)
libraryDependencies += "com.github.scopt" %% "scopt" % "3.5.0"

releaseSettings

scalariformSettings

ivyScala := ivyScala.value map { _.copy(overrideScalaVersion = true) }

//routesGenerator := InjectedRoutesGenerator

/*
mappings in Universal ++=
  (baseDirectory.value / "ML_models" * "*" get) map
    (x => x -> ("ML_models/" + x.getName))
*/

//mappings in Universal ++= directory(baseDirectory.value / "ML_models")

fork in run := true