package be.guldentops.geert.simple.ml.normalization;

import org.ejml.simple.SimpleMatrix;

public class Normalizer {

    public SimpleMatrix calculateMean(SimpleMatrix matrix) {
        SimpleMatrix mean = new SimpleMatrix(1, matrix.numCols());

        for (int j = 0; j < matrix.numCols(); j++) {
            double columnMean = 0.0;

            for (int i = 0; i < matrix.numRows(); i++) {
                columnMean += matrix.get(i, j);
            }

            mean.set(0, j, columnMean / matrix.numRows());
        }

        return mean;
    }

    public SimpleMatrix calculateStandardDeviation(SimpleMatrix matrix, SimpleMatrix mean) {
        SimpleMatrix standardDeviation = new SimpleMatrix(1, matrix.numCols());

        for (int j = 0; j < matrix.numCols(); j++) {
            double columnStandardDeviation = 0.0;
            double columnMean = mean.get(0, j);

            for (int i = 0; i < matrix.numRows(); i++) {
                columnStandardDeviation += Math.pow(matrix.get(i, j) - columnMean, 2);
            }

            standardDeviation.set(0, j, Math.sqrt(columnStandardDeviation / (matrix.numRows() - 1)));
        }

        return standardDeviation;
    }

    public SimpleMatrix normalize(SimpleMatrix matrix, SimpleMatrix mean, SimpleMatrix standardDeviation) {
        SimpleMatrix normalizedMatrix = new SimpleMatrix(matrix.numRows(), matrix.numCols());

        for (int j = 0; j < matrix.numCols(); j++) {
            double columnMean = mean.get(0, j);
            double columnStandardDeviation = standardDeviation.get(0, j);

            for (int i = 0; i < matrix.numRows(); i++) {
                normalizedMatrix.set(i, j, ((matrix.get(i, j) - columnMean) / handleZero(columnStandardDeviation)));
            }
        }

        return normalizedMatrix;
    }

    private double handleZero(double columnStandardDeviation) {
        // Division by zero results in NaN -> divide by 1 instead since 1 will have no effect
        return columnStandardDeviation != 0 ? columnStandardDeviation : 1;
    }
}
