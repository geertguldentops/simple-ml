package be.guldentops.geert.simple.ml.logistic.regression;

import be.guldentops.geert.simple.ml.Hyperparameters;
import org.ejml.simple.SimpleMatrix;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.ones;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.zeros;

public class MultivariateLogisticRegression implements LogisticRegression {

    private final Hyperparameters hyperparameters;

    private SimpleMatrix features;
    private SimpleMatrix labels;
    private SimpleMatrix model;

    public MultivariateLogisticRegression(Hyperparameters hyperparameters) {
        this.hyperparameters = hyperparameters;
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
        this.model = gradientDescent(applyBias(features), labels);
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

    /* default */SimpleMatrix costFunction(SimpleMatrix features, SimpleMatrix labels, SimpleMatrix theta) {
        var m = features.numRows();
        var biasedFeatures = applyBias(features);

        var hypothesis = sigmoid(biasedFeatures.mult(theta));
        var derivedCostFunction = (biasedFeatures.transpose()).mult(hypothesis.minus(labels));

        return derivedCostFunction.divide(m);
    }

    private SimpleMatrix gradientDescent(SimpleMatrix features, SimpleMatrix labels) {
        var m = features.numRows();
        var theta = initialiseTheta(features.numCols());

        for (int i = 0; i < hyperparameters.maxIterations(); i++) {
            var h = sigmoid(features.mult(theta));
            var derivedCostFunction = (features.transpose()).mult(h.minus(labels));

            // TODO: Cost function doesn't seem to work with learning rate/gradient descent
            theta = theta.minus(derivedCostFunction.scale(hyperparameters.learningRate() / m));
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
        var biasedNewData = applyBias(newData);

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
}
