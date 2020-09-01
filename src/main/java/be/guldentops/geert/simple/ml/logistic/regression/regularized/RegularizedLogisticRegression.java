package be.guldentops.geert.simple.ml.logistic.regression.regularized;

import be.guldentops.geert.simple.ml.logistic.regression.LogisticRegression;
import be.guldentops.geert.simple.ml.normalization.Normalizer;
import org.ejml.simple.SimpleMatrix;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.ones;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.zeros;

public class RegularizedLogisticRegression implements LogisticRegression {

    private final Hyperparameters hyperparameters;
    private final Normalizer normalizer;

    private SimpleMatrix features;
    private SimpleMatrix labels;

    private SimpleMatrix mean;
    private SimpleMatrix standardDeviation;

    private SimpleMatrix model;

    public RegularizedLogisticRegression(Hyperparameters hyperparameters) {
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
        this.features = addPolynomialFeatures(extractFeatures(trainingSet));
        this.labels = extractLabels(trainingSet);

        this.mean = normalizer.calculateMean(features);
        this.standardDeviation = normalizer.calculateStandardDeviation(features, mean);

        this.model = gradientDescent(applyBias(normalizer.normalize(features, mean, standardDeviation)), labels);
    }

    private SimpleMatrix extractFeatures(SimpleMatrix trainingSet) {
        return trainingSet.cols(0, trainingSet.numCols() - 1);
    }

    private SimpleMatrix addPolynomialFeatures(SimpleMatrix features) {
        SimpleMatrix X1 = features.extractVector(false, 0);
        SimpleMatrix X2 = features.extractVector(false, 1);

        SimpleMatrix X = new SimpleMatrix(features.numRows(), 27);

        int col = 0;
        for (int i = 1; i <= 6; i++) {
            for (int j = 0; j <= i; j++) {
                X.insertIntoThis(0, col, (X1.elementPower(i - j)).elementMult(X2.elementPower(j)));
                col++;
            }
        }

        return X;
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
            SimpleMatrix regularizationTerm = thetaWithoutBias(theta).scale(1 - (hyperparameters.getLearningRate() * hyperparameters.getLambda() / m));

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

        return sigmoid(biasedNewData.mult(model));
    }

    @Override
    public SimpleMatrix predictMany(SimpleMatrix newData) {
        SimpleMatrix predictions = predict(newData);

        for (int i = 0; i < predictions.numRows(); i++) {
            predictions.set(i, 0, predictions.get(i, 0) >= 0.5 ? 1 : 0);
        }

        return predictions;
    }

    public static final class Hyperparameters {

        private final double learningRate;
        private final int maxIterations;
        private final double lambda;

        public Hyperparameters(double learningRate, int maxIterations, double lambda) {
            this.learningRate = learningRate;
            this.maxIterations = maxIterations;
            this.lambda = lambda;
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
    }
}
