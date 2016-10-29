package ml

/**
 * Created by abderrahmen on 29/10/2016.
 */




object LogisticRegression {


  case class Params(
                     input: String = null,
                     testInput: String = "",
                     dataFormat: String = "libsvm",
                     regParam: Double = 0.0,
                     elasticNetParam: Double = 0.0,
                     maxIter: Int = 100,
                     fitIntercept: Boolean = true,
                     tol: Double = 1E-6,
                     fracTest: Double = 0.2)


}
