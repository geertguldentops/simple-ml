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

    /* default */SimpleMatrix model() {
        return model;
    }

    @Override
    public void learn(SimpleMatrix trainingSet) {
        this.features = extractFeatures(trainingSet);
        this.labels = extractLabels(trainingSet);

        this.mean = normalizer.calculateMean(features);
        this.standardDeviation = normalizer.calculateStandardDeviation(features, mean);

        // Add + 1 to account for bias
        var allThetas = new SimpleMatrix(hyperparameters.numberOfLabels, features.numCols() + 1);
        allThetas.zero();

        for (int i = 0; i < hyperparameters.numberOfLabels; i++) {
            var classGroundTruth = eq(labels, i + 1);
            System.out.printf("Staring calculation of gradient descent for number %d (zero == 10)\n", (i + 1));
            var theta = gradientDescent(applyBias(normalizer.normalize(features, mean, standardDeviation)), classGroundTruth);
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
            // It should be "thetaWithoutBias(theta)" but training accuracy drops by 21% if we do that!
            var regularizationTerm = /*thetaWithoutBias(*/theta/*)*/.scale(1 - (hyperparameters.learningRate * hyperparameters.lambda / m));

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

        return sigmoid(biasedNewData.mult(model.transpose()));
    }

    @Override
    public SimpleMatrix predictMany(SimpleMatrix newData) {
        var predictions = predict(newData);

        var finalPredictions = new SimpleMatrix(newData.numRows(), 1);
        for (int i = 0; i < predictions.numRows(); i++) {
            var maxIndex = maxIndex(predictions.extractVector(true, i));
            finalPredictions.set(i, 0, (maxIndex + 1)); // Data is 1-indexed instead of 0-indexed!
        }

        return finalPredictions;
    }

    public record Hyperparameters(double learningRate,
                                  int maxIterations,
                                  double lambda,
                                  int numberOfLabels) {
    }
}
