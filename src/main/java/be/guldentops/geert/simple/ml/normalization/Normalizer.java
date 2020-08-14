package be.guldentops.geert.simple.ml.normalization;

import org.ejml.simple.SimpleMatrix;

public class Normalizer {

    public SimpleMatrix calculateMean(SimpleMatrix matrix) {
        var mean = new SimpleMatrix(1, matrix.numCols());

        for (int j = 0; j < matrix.numCols(); j++) {
            var columnMean = 0.0;

            for (int i = 0; i < matrix.numRows(); i++) {
                columnMean += matrix.get(i, j);
            }

            mean.set(0, j, columnMean / matrix.numRows());
        }

        return mean;
    }

    public SimpleMatrix calculateStandardDeviation(SimpleMatrix matrix, SimpleMatrix mean) {
        var standardDeviation = new SimpleMatrix(1, matrix.numCols());

        for (int j = 0; j < matrix.numCols(); j++) {
            var columnStandardDeviation = 0.0;
            var columnMean = mean.get(0, j);

            for (int i = 0; i < matrix.numRows(); i++) {
                columnStandardDeviation += Math.pow(matrix.get(i, j) - columnMean, 2);
            }

            standardDeviation.set(0, j, Math.sqrt(columnStandardDeviation / (matrix.numRows() - 1)));
        }

        return standardDeviation;
    }

    public SimpleMatrix normalize(SimpleMatrix matrix, SimpleMatrix mean, SimpleMatrix standardDeviation) {
        var normalizedMatrix = new SimpleMatrix(matrix.numRows(), matrix.numCols());

        for (int j = 0; j < matrix.numCols(); j++) {
            var columnMean = mean.get(0, j);
            var columnStandardDeviation = standardDeviation.get(0, j);

            for (int i = 0; i < matrix.numRows(); i++) {
                normalizedMatrix.set(i, j, ((matrix.get(i, j) - columnMean) / columnStandardDeviation));
            }
        }

        return normalizedMatrix;
    }
}
