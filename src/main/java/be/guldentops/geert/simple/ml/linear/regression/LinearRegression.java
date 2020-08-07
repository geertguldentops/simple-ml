package be.guldentops.geert.simple.ml.linear.regression;

import org.ejml.data.Matrix;

public interface LinearRegression {

	void learn(Matrix trainingSet);

	double predict(Matrix newData);

}
