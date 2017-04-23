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
        var conf = new SparkConf().setAppName("DecisionTreeRegressionExample")
        var sc = new SparkContext(conf)

        def stringFilterHash(a : String) : String = {
            if(a.equalsIgnoreCase("?")) return "?"
            return a.hashCode().toString()
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

        def filterBirdie(a : String) : String = {
            if(a.equalsIgnoreCase("X")) return "1";
            if(a.toDouble > 0) return "1";
            return "0"
        }


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


            return Vectors.sparse(a.size, x.toArray, y.toArray)
        }

        if(args(0).equalsIgnoreCase("--predict")){
            var csv: RDD[Seq[String]] = sc.textFile(args(1)).map(_.split(","));
            csv = csv.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
            var csvFeaturesAndLabels = csv.map(a => (List(a(26),
                //test
                stringFilterHash(a(1)), stringFilterHash(a(8)),
                stringFilterHash(a(9)), stringFilterHash(a(10)),
                stringFilterHash(a(11)), stringFilterHash(a(15)),
                stringFilterHash(a(18)), stringFilterHash(a(961)),
                //end test
                a(2), a(3), a(4), a(5), a(6), a(7),
                a(12), a(13), a(14),
                a(16),
                a(955), a(958), a(957), a(956), a(959), a(960),
                a(962), a(963), a(964), a(965), a(966), a(967),
                filterBirdie(a(320)),
                a(1090), a(1091), a(1092), a(1093), a(1094),
                a(1095), a(1096), a(1097), a(1098), a(1099), a(1100),
                a(1101)), a(0)))


            val actionableVectors = csvFeaturesAndLabels.map(a => (a._2, genVector(a._1.slice(1, a._1.size - 1))))
            val model = RandomForestModel.load(sc, args(2) + "2/output.tree")
            var finalPred = actionableVectors.map(a => (a._1, model.predict(a._2))).map(a => {
                a._1 + "," + a._2.toString()
            })
            var finalOutToWritePre = finalPred.coalesce(1, false)

            var finalOutToWrite = sc.parallelize(List("SAMPLING_EVENT_ID, SAW_AGELAIUS_PHOENICEUS \n")) ++
                    finalOutToWritePre
            finalOutToWrite = finalOutToWrite.coalesce(1, false)

            finalOutToWrite.saveAsTextFile(args(2) + "/output.predictions")
        }
        else {

            var csv: RDD[Seq[String]] = sc.textFile(args(1)).map(_.split(","));
            csv = csv.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
            var csvFeatures = csv.map(a => List(a(26),
                //test
                stringFilterHash(a(1)), stringFilterHash(a(8)),
                stringFilterHash(a(9)), stringFilterHash(a(10)),
                stringFilterHash(a(11)), stringFilterHash(a(15)),
                stringFilterHash(a(18)), stringFilterHash(a(961)),
                //end test
                a(2), a(3), a(4), a(5), a(6), a(7),
                a(12), a(13), a(14),
                a(16),
                a(955), a(958), a(957), a(956), a(959), a(960),
                a(962), a(963), a(964), a(965), a(966), a(967),
                filterBirdie(a(320)),
                a(1090), a(1091), a(1092), a(1093), a(1094),
                a(1095), a(1096), a(1097), a(1098), a(1099), a(1100),
                a(1101)))

            val transformedCsv = csvFeatures.
                    filter(a => (!a(0).equals("?"))).
                    map(a => LabeledPoint(binaryPart(a(0)),
                        genVector(a.slice(1, a.size - 1))))

            System.out.println(System.currentTimeMillis()/1000 + ":0:transformedCsv")
            val numClasses = 2
            val categoricalFeaturesInfo = Map[Int, Int]()
            val numTrees = 50
            // Use more in practice.
            val featureSubsetStrategy = "auto"
            // Let the algorithm choose.
            val impurity = "gini"
            val maxDepth = 15
            val maxBins = 32

            val temp = transformedCsv.randomSplit(Array(0.7, 0.3))
            val (trainingData, testData) = (temp(0), temp(1))
            trainingData.persist()
            testData.persist()

            System.out.println(System.currentTimeMillis()/1000 + ":1:transformedCsv")
            //model is not an RDD, it is not re-calculated.

            val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
                numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
            System.out.println(System.currentTimeMillis()/1000 + ":trained")

            val labelAndPreds = testData.map { point =>
                val prediction = model.predict(point.features)
                (point.label, prediction)
            }
            System.out.println(System.currentTimeMillis()/1000 + ":predicted")

            val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
            model.save(sc, args(2) + "2/output.tree")
            System.out.println("Accuracy = " + (1 - testErr) * 100)
            System.out.println("Learned classification forest model:\n" + model.toDebugString)
        }
        sc.stop()

    }

}

