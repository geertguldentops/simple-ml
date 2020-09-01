package be.guldentops.geert.simple.ml.linear.regression.multivariate;

import be.guldentops.geert.simple.ml.linear.regression.LinearRegression;
import be.guldentops.geert.simple.ml.normalization.Normalizer;
import org.ejml.simple.SimpleMatrix;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.ones;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.zeros;

public class MultivariateLinearRegression implements LinearRegression {

    private final Hyperparameters hyperparameters;
    private final Normalizer normalizer;

    private SimpleMatrix mean;
    private SimpleMatrix standardDeviation;

    private SimpleMatrix model;

    public MultivariateLinearRegression(Hyperparameters hyperparameters) {
        this.hyperparameters = hyperparameters;
        this.normalizer = new Normalizer();
    }

    @Override
    public void learn(SimpleMatrix trainingSet) {
        SimpleMatrix features = extractFeatures(trainingSet);
        SimpleMatrix labels = extractLabels(trainingSet);

        this.mean = normalizer.calculateMean(features);
        this.standardDeviation = normalizer.calculateStandardDeviation(features, mean);

        this.model = gradientDescent(applyBias(normalizer.normalize(features, mean, standardDeviation)), labels);
    }

    private SimpleMatrix extractFeatures(SimpleMatrix trainingSet) {
        return trainingSet.cols(0, trainingSet.numCols() - 1);
    }

    private SimpleMatrix extractLabels(SimpleMatrix trainingSet) {
        return trainingSet.cols(trainingSet.numCols() - 1, trainingSet.numCols());
    }

    private SimpleMatrix applyBias(SimpleMatrix features) {
        int m = features.numRows();
        SimpleMatrix bias = ones(m);

        SimpleMatrix biasedFeatures = new SimpleMatrix(m, features.numCols() + 1);
        biasedFeatures.insertIntoThis(0, 0, bias);
        biasedFeatures.insertIntoThis(0, 1, features);

        return biasedFeatures;
    }

    private SimpleMatrix gradientDescent(SimpleMatrix features, SimpleMatrix labels) {
        int m = features.numRows();
        SimpleMatrix theta = initialiseTheta(features.numCols());

        for (int i = 0; i < hyperparameters.getMaxIterations(); i++) {
            SimpleMatrix h = features.mult(theta);
            SimpleMatrix derivedCostFunction = features.transpose().mult(h.minus(labels));

            theta = theta.minus(derivedCostFunction.scale(hyperparameters.getLearningRate() / m));
        }

        return theta;
    }

    private SimpleMatrix initialiseTheta(int n) {
        return zeros(n);
    }

    @Override
    public double predict(SimpleMatrix newData) {
        SimpleMatrix prediction = applyBias(normalizer.normalize(newData, mean, standardDeviation)).mult(model);

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
