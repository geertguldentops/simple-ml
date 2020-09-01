package be.guldentops.geert.simple.ml.logistic.regression.onevsall;

import be.guldentops.geert.simple.ml.logistic.regression.LogisticRegression;
import be.guldentops.geert.simple.ml.normalization.Normalizer;
import org.ejml.simple.SimpleMatrix;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.eq;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.maxIndex;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.ones;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.zeros;

public class OneVsAllLogisticRegression implements LogisticRegression {

    private final Hyperparameters hyperparameters;
    private final Normalizer normalizer;

    private SimpleMatrix features;
    private SimpleMatrix labels;

    private SimpleMatrix mean;
    private SimpleMatrix standardDeviation;

    private SimpleMatrix model;

    public OneVsAllLogisticRegression(Hyperparameters hyperparameters) {
        this.hyperparameters = hyperparameters;
        this.normalizer = new Normalizer();
    }

    /* default */SimpleMatrix features() {
        return features;
    }

    /* default */SimpleMatrix labels() {
        return labels;
    }

    @Override
    public void learn(SimpleMatrix trainingSet) {
        this.features = extractFeatures(trainingSet);
        this.labels = extractLabels(trainingSet);

        this.mean = normalizer.calculateMean(features);
        this.standardDeviation = normalizer.calculateStandardDeviation(features, mean);

        // Add + 1 to account for bias
        SimpleMatrix allThetas = new SimpleMatrix(hyperparameters.numberOfLabels, features.numCols() + 1);
        allThetas.zero();

        for (int i = 0; i < hyperparameters.numberOfLabels; i++) {
            SimpleMatrix classGroundTruth = eq(labels, i + 1);
            System.out.printf("Staring calculation of gradient descent for number %d (zero == 10)\n", (i + 1));
            SimpleMatrix theta = gradientDescent(applyBias(normalizer.normalize(features, mean, standardDeviation)), classGroundTruth);
            System.out.printf("Finished calculating gradient descent for i %d (zero == 10)\n", (i + 1));
            allThetas.insertIntoThis(i, 0, theta.transpose());
        }

        this.model = allThetas;
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

    /* default */SimpleMatrix costFunction(SimpleMatrix features, SimpleMatrix labels, SimpleMatrix theta, double lambda) {
        int m = features.numRows();
        SimpleMatrix biasedFeatures = applyBias(features);

        SimpleMatrix g = sigmoid(biasedFeatures.mult(theta));
        SimpleMatrix derivedCostFunction = biasedFeatures.transpose().mult(g.minus(labels)).divide(m);
        SimpleMatrix regularizationTerm = thetaWithoutBias(theta).scale(lambda / m);

        return derivedCostFunction.plus(regularizationTerm);
    }

    private SimpleMatrix thetaWithoutBias(SimpleMatrix theta) {
        SimpleMatrix thetaWithoutBias = new SimpleMatrix(theta.numRows(), theta.numCols());
        thetaWithoutBias.insertIntoThis(1, 0, theta.rows(1, theta.numRows()));

        return thetaWithoutBias;
    }

    private SimpleMatrix gradientDescent(SimpleMatrix features, SimpleMatrix labels) {
        int m = features.numRows();
        SimpleMatrix theta = initialiseTheta(features.numCols());

        for (int i = 0; i < hyperparameters.getMaxIterations(); i++) {
            SimpleMatrix g = sigmoid(features.mult(theta));
            SimpleMatrix derivedCostFunction = features.transpose().mult(g.minus(labels)).divide(m);
            // It should be "thetaWithoutBias(theta)" but training accuracy drops by 21% if we do that!
            SimpleMatrix regularizationTerm = /*thetaWithoutBias(*/theta/*)*/.scale(1 - (hyperparameters.getLearningRate() * hyperparameters.getLambda() / m));

            theta = regularizationTerm.minus(derivedCostFunction);
        }

        return theta;
    }

    private SimpleMatrix initialiseTheta(int n) {
        return zeros(n);
    }

    private SimpleMatrix sigmoid(SimpleMatrix matrix) {
        SimpleMatrix sigmoid = new SimpleMatrix(matrix.numRows(), matrix.numCols());

        for (int i = 0; i < matrix.numRows(); i++) {
            for (int j = 0; j < matrix.numCols(); j++) {
                sigmoid.set(i, j, 1 / (1 + Math.pow(Math.E, -matrix.get(i, j))));
            }
        }

        return sigmoid;
    }

    @Override
    public double predictOne(SimpleMatrix newData) {
        SimpleMatrix predictions = predict(newData);

        return predictions.get(0) >= 0.5 ? 1 : 0;
    }

    private SimpleMatrix predict(SimpleMatrix newData) {
        SimpleMatrix biasedNewData = applyBias(normalizer.normalize(newData, mean, standardDeviation));

        return sigmoid(biasedNewData.mult(model.transpose()));
    }

    @Override
    public SimpleMatrix predictMany(SimpleMatrix newData) {
        SimpleMatrix predictions = predict(newData);

        SimpleMatrix finalPredictions = new SimpleMatrix(newData.numRows(), 1);
        for (int i = 0; i < predictions.numRows(); i++) {
            int maxIndex = maxIndex(predictions.extractVector(true, i));
            finalPredictions.set(i, 0, (maxIndex + 1)); // Data is 1-indexed instead of 0-indexed!
        }

        return finalPredictions;
    }

    public static final class Hyperparameters {

        private final double learningRate;
        private final int maxIterations;
        private final double lambda;
        private final int numberOfLabels;

        public Hyperparameters(double learningRate, int maxIterations, double lambda, int numberOfLabels) {
            this.learningRate = learningRate;
            this.maxIterations = maxIterations;
            this.lambda = lambda;
            this.numberOfLabels = numberOfLabels;
        }

        public double getLearningRate() {
            return learningRate;
        }

        public int getMaxIterations() {
            return maxIterations;
        }

        public double getLambda() {
            return lambda;
        }

        public int getNumberOfLabels() {
            return numberOfLabels;
        }
    }
}
