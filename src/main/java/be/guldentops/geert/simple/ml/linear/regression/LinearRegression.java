package be.guldentops.geert.simple.ml.linear.regression;

import org.ejml.simple.SimpleMatrix;

public interface LinearRegression {

	void learn(SimpleMatrix trainingSet);

	double predict(SimpleMatrix newData);

}
