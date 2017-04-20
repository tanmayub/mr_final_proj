/**
  * Created by surana on 4/17/17.
  */
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors

import scala.collection.mutable.{ArrayBuffer, ListBuffer}


object aa {

  def main(args : Array[String]) : Unit  = {



    def parsePeople(a : String) : String = {
      if(a.equalsIgnoreCase("?")) return "2";
      return (((a.toDouble)/3)).toInt.toString;
    };


    def parseEffort(a : String) : String = {
      if(a.equalsIgnoreCase("?")) return 10.toString;
      return (((a.toDouble) * 100)/10).toInt.toString;
    };


    def timeParse(a : String) : String = {
      if(a.equalsIgnoreCase("?")) return 0.toString;
      return ((a.toDouble)/4).toInt.toString;
    };


    val hashingTF = new HashingTF(18);

    def binaryPart(a: String): Double = {
      if (a.equalsIgnoreCase("X") || (a.toDouble > 0)) return 1.0
      return 0.0
    }

    var conf = new SparkConf().setAppName("DecisionTreeRegressionExample").setMaster("local[*]")
    var sc = new SparkContext(conf)
    var csv: RDD[Seq[String]] = sc.textFile("input/xaa.csv").map(_.split(","));
    csv = csv.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    var csvFeatures = csv.map(a => List(a(26),
      a(2), a(3),
      a(5), a(7), a(6),
      a(12), a(13), a(14),
      a(16),
      a(955), a(958), a(957), a(956),
      a(964), a(965),
      a(967)));


    def genVector(a : Seq[String]): org.apache.spark.mllib.linalg.Vector ={
        var x = ArrayBuffer[Int]()
        var y = ArrayBuffer[Double]();
        a.zipWithIndex.
          map{
            case (i, j) => if(!i.equalsIgnoreCase("?"))
              {x.append(j) ; y.append(i.toDouble)}
          }
        if(x.size == 0 || y.size == 0)
            System.out.println(a.toString())


        return Vectors.sparse(20, x.toArray, y.toArray)
    }

    val transformedCsv = csvFeatures.
      filter(a => (!a(0).equals("?"))).
      map(a => LabeledPoint(binaryPart(a(0)),
        genVector(a.slice(1, a.size - 1))))


    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 12
    // Use more in practice.
    val featureSubsetStrategy = "auto"
    // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 16
    val maxBins = 32

    val temp = transformedCsv.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (temp(0), temp(1))
    trainingData.cache()

    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    System.out.println("Accuracy = " + (1 - testErr) * 100)
    System.out.println("Learned classification forest model:\n" + model.toDebugString)
  }

}

