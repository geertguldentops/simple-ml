package be.guldentops.geert.simple.ml.linear.regression.univariate;

import be.guldentops.geert.simple.ml.linear.regression.LinearRegression;
import org.ejml.data.Matrix;
import org.ejml.simple.SimpleMatrix;

import static be.guldentops.geert.simple.ml.ArrayUtilities.ones;

public class UnivariateLinearRegression implements LinearRegression {

	private SimpleMatrix features;
	private SimpleMatrix labels;

	/**
	 * Returns the feature matrix, including the bias
	 *
	 * @return the feature matrix (with dimensions [# rows, 2])
	 */
	/* default */SimpleMatrix features() {
		return features;
	}

	/**
	 * Returns the label column vector
	 *
	 * @return the label column vector (with dimensions [# rows, 1])
	 */
	/* default */SimpleMatrix labels() {
		return labels;
	}

	@Override
	public void learn(Matrix trainingSet) {
		var features = SimpleMatrix.wrap(trainingSet).cols(0, 1);
		var m = features.getMatrix().getNumRows();

		this.features = applyBias(m, features);
		this.labels = SimpleMatrix.wrap(trainingSet).cols(1, 2);
	}

	private SimpleMatrix applyBias(int m, SimpleMatrix features) {
		var bias = ones(m);

		var biasedFeatures = new SimpleMatrix(m, 2);
		biasedFeatures.setColumn(0, 0, bias);
		biasedFeatures.insertIntoThis(0, 1, features);

		return biasedFeatures;
	}

	@Override
	public double predict(Matrix newData) {
		throw new UnsupportedOperationException("Not implemented!");
	}
}
