package be.guldentops.geert.simple.ml.linear.regression.univariate;

import be.guldentops.geert.simple.ml.Hyperparameters;
import be.guldentops.geert.simple.ml.linear.regression.LinearRegression;
import org.ejml.data.Matrix;
import org.ejml.simple.SimpleMatrix;

import static be.guldentops.geert.simple.ml.ArrayUtilities.ones;
import static be.guldentops.geert.simple.ml.ArrayUtilities.zeros;

public class UnivariateLinearRegression implements LinearRegression {

	private final Hyperparameters hyperparameters;

	private SimpleMatrix features;
	private SimpleMatrix labels;
	private SimpleMatrix model;

	public UnivariateLinearRegression(Hyperparameters hyperparameters) {
		this.hyperparameters = hyperparameters;
	}

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

	/**
	 * Returns the model used by the ML algorithm to make predictions
	 *
	 * @return the ML model column vector (with dimensions [2, 1])
	 */
	/* default */ SimpleMatrix model() {
		return model;
	}

	@Override
	public void learn(Matrix trainingSet) {
		this.features = applyBias(extractFeatures(trainingSet));
		this.labels = extractLabels(trainingSet);
		this.model = gradientDescent(features, labels);
	}

	private SimpleMatrix extractFeatures(Matrix trainingSet) {
		return SimpleMatrix.wrap(trainingSet).cols(0, 1);
	}

	private SimpleMatrix applyBias(SimpleMatrix features) {
		var m = sizeOf(features);
		var bias = ones(m);

		var biasedFeatures = new SimpleMatrix(m, 2);
		biasedFeatures.setColumn(0, 0, bias);
		biasedFeatures.insertIntoThis(0, 1, features);

		return biasedFeatures;
	}

	private int sizeOf(SimpleMatrix features) {
		return features.getMatrix().getNumRows();
	}

	private SimpleMatrix extractLabels(Matrix trainingSet) {
		return SimpleMatrix.wrap(trainingSet).cols(1, 2);
	}

	private SimpleMatrix gradientDescent(SimpleMatrix features, SimpleMatrix labels) {
		var m = sizeOf(features);
		var theta = initialiseTheta();

		for (int i = 0; i < hyperparameters.maxIterations(); i++) {
			var h = features.mult(theta);
			var derivedCostFunction = (features.transpose()).mult(h.minus(labels));

			theta = theta.minus(derivedCostFunction.scale(hyperparameters.learningRate() / m));
		}

		return theta;
	}

	private SimpleMatrix initialiseTheta() {
		return new SimpleMatrix(2, 1, true, zeros(2));
	}

	@Override
	public double predict(Matrix newData) {
		throw new UnsupportedOperationException("Not implemented!");
	}
}
