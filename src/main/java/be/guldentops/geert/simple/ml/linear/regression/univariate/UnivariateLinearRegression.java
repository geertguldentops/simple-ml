package be.guldentops.geert.simple.ml.linear.regression.univariate;

import be.guldentops.geert.simple.ml.linear.regression.LinearRegression;
import org.ejml.simple.SimpleMatrix;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.ones;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.zeros;

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
    public void learn(SimpleMatrix trainingSet) {
        this.features = applyBias(extractFeatures(trainingSet));
        this.labels = extractLabels(trainingSet);
        this.model = gradientDescent(features, labels);
    }

    private SimpleMatrix extractFeatures(SimpleMatrix trainingSet) {
        return trainingSet.cols(0, 1);
    }

    private SimpleMatrix applyBias(SimpleMatrix features) {
        int m = features.numRows();
        SimpleMatrix bias = ones(m);

        SimpleMatrix biasedFeatures = new SimpleMatrix(m, 2);
        biasedFeatures.insertIntoThis(0, 0, bias);
        biasedFeatures.insertIntoThis(0, 1, features);

        return biasedFeatures;
    }

    private SimpleMatrix extractLabels(SimpleMatrix trainingSet) {
        return trainingSet.cols(1, 2);
    }

    private SimpleMatrix gradientDescent(SimpleMatrix features, SimpleMatrix labels) {
        int m = features.numRows();
        SimpleMatrix theta = initialiseTheta(features.numCols());

        for (int i = 0; i < hyperparameters.getMaxIterations(); i++) {
            SimpleMatrix h = features.mult(theta);
            SimpleMatrix derivedCostFunction = (features.transpose()).mult(h.minus(labels));

            theta = theta.minus(derivedCostFunction.scale(hyperparameters.getLearningRate() / m));
        }

        return theta;
    }

    private SimpleMatrix initialiseTheta(int n) {
        return zeros(n);
    }

    @Override
    public double predict(SimpleMatrix newData) {
        SimpleMatrix biasedNewData = applyBias(newData);
        SimpleMatrix prediction = biasedNewData.mult(model);

        return prediction.get(0);
    }

    public static final class Hyperparameters {

        private final double learningRate;
        private final int maxIterations;

        public Hyperparameters(double learningRate, int maxIterations) {
            this.learningRate = learningRate;
            this.maxIterations = maxIterations;
        }

        public double getLearningRate() {
            return learningRate;
        }

        public int getMaxIterations() {
            return maxIterations;
        }
    }
}
