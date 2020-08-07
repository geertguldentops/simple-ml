package be.guldentops.geert.simple.ml.linear.regression.univariate;

import be.guldentops.geert.simple.ml.linear.regression.LinearRegression;
import org.ejml.data.Matrix;
import org.ejml.simple.SimpleMatrix;

import static be.guldentops.geert.simple.ml.ArrayUtilities.ones;

public class UnivariateLinearRegression implements LinearRegression {

	private SimpleMatrix X;

	/**
	 * Returns X, the feature matrix, including the bias
	 *
	 * @return the feature matrix X, with dimensions [# rows, 2]
	 */
	/* default */SimpleMatrix X() {
		return X;
	}

	@Override
	public void learn(Matrix trainingSet) {
		var feature = SimpleMatrix.wrap(trainingSet).cols(0, 1);
		var m = feature.getMatrix().getNumRows();

		this.X = applyBias(m, feature);
	}

	private SimpleMatrix applyBias(int m, SimpleMatrix features) {
		var biasedX = new SimpleMatrix(m, 2);
		biasedX.setColumn(0, 0, ones(m));
		biasedX.insertIntoThis(0, 1, features);

		return biasedX;
	}

	@Override
	public double predict(Matrix newData) {
		throw new UnsupportedOperationException("Not implemented!");
	}
}
