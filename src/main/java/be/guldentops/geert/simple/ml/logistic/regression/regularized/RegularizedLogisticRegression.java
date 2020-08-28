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

    /* default */SimpleMatrix model() {
        return model;
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
        var X1 = features.extractVector(false, 0);
        var X2 = features.extractVector(false, 1);

        var X = new SimpleMatrix(features.numRows(), 27);

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
        var m = features.numRows();
        var bias = ones(m);

        var biasedFeatures = new SimpleMatrix(m, features.numCols() + 1);
        biasedFeatures.insertIntoThis(0, 0, bias);
        biasedFeatures.insertIntoThis(0, 1, features);

        return biasedFeatures;
    }

    /* default */SimpleMatrix costFunction(SimpleMatrix features, SimpleMatrix labels, SimpleMatrix theta, double lambda) {
        var m = features.numRows();
        var biasedFeatures = applyBias(features);

        var g = sigmoid(biasedFeatures.mult(theta));
        var derivedCostFunction = biasedFeatures.transpose().mult(g.minus(labels)).divide(m);
        var regularizationTerm = thetaWithoutBias(theta).scale(lambda / m);

        return derivedCostFunction.plus(regularizationTerm);
    }

    private SimpleMatrix thetaWithoutBias(SimpleMatrix theta) {
        var thetaWithoutBias = new SimpleMatrix(theta.numRows(), theta.numCols());
        thetaWithoutBias.insertIntoThis(1, 0, theta.rows(1, theta.numRows()));

        return thetaWithoutBias;
    }

    private SimpleMatrix gradientDescent(SimpleMatrix features, SimpleMatrix labels) {
        var m = features.numRows();
        var theta = initialiseTheta(features.numCols());

        for (int i = 0; i < hyperparameters.maxIterations(); i++) {
            var g = sigmoid(features.mult(theta));
            var derivedCostFunction = features.transpose().mult(g.minus(labels)).divide(m);
            var regularizationTerm = thetaWithoutBias(theta).scale(1 - (hyperparameters.learningRate * hyperparameters.lambda / m));

            theta = regularizationTerm.minus(derivedCostFunction);
        }

        return theta;
    }

    private SimpleMatrix initialiseTheta(int n) {
        return zeros(n);
    }

    private SimpleMatrix sigmoid(SimpleMatrix matrix) {
        var sigmoid = new SimpleMatrix(matrix.numRows(), matrix.numCols());

        for (int i = 0; i < matrix.numRows(); i++) {
            for (int j = 0; j < matrix.numCols(); j++) {
                sigmoid.set(i, j, 1 / (1 + Math.pow(Math.E, -matrix.get(i, j))));
            }
        }

        return sigmoid;
    }

    @Override
    public double predictOne(SimpleMatrix newData) {
        var predictions = predict(newData);

        return predictions.get(0) >= 0.5 ? 1 : 0;
    }

    private SimpleMatrix predict(SimpleMatrix newData) {
        var biasedNewData = applyBias(normalizer.normalize(newData, mean, standardDeviation));

        return sigmoid(biasedNewData.mult(model));
    }

    @Override
    public SimpleMatrix predictMany(SimpleMatrix newData) {
        var predictions = predict(newData);

        for (int i = 0; i < predictions.numRows(); i++) {
            predictions.set(i, 0, predictions.get(i, 0) >= 0.5 ? 1 : 0);
        }

        return predictions;
    }

    public record Hyperparameters(double learningRate, int maxIterations, double lambda) {
    }
}
