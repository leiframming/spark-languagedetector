val libDeps = Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "2.2.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "2.2.0" % "provided",
  "org.specs2" % "specs2_2.11" % "3.7" % "test" pomOnly()
)



lazy val common = Project("spark-languagedetector", file("."))
  .configs(IntegrationTest)
  .settings(
    name := "scala-languagedetector",
    scalaVersion := "2.11.11",
    organization := "lbl",
    libraryDependencies ++= libDeps,
    Defaults.itSettings
  )
