val libDeps = Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "2.2.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "2.2.0" % "provided",
  "org.specs2" % "specs2_2.11" % "3.7" % "test" pomOnly(),
  "com.typesafe" % "config" % "1.3.1" % "test"
)



lazy val common = Project("spark-languagedetector", file("."))
    .enablePlugins(GitVersioning)
  .configs(IntegrationTest)
  .settings(
    name := "spark-languagedetector",
    scalaVersion := "2.11.11",
    organization := "lbl",
    libraryDependencies ++= libDeps,
    Defaults.itSettings,
    git.useGitDescribe := true
  )
