package be.guldentops.geert.simple.ml;

import org.ejml.simple.SimpleMatrix;

public final class SimpleMatrixUtilities {

    private SimpleMatrixUtilities() {
    }

    public static SimpleMatrix zeros(int numRows) {
        var columnVector = new SimpleMatrix(numRows, 1);
        columnVector.zero();

        return columnVector;
    }

    public static SimpleMatrix ones(int numRows) {
        var columnVector = new SimpleMatrix(numRows, 1);
        columnVector.fill(1.0);

        return columnVector;
    }

    public static double mean(SimpleMatrix a) {
        var mean = 0.0;

        var n = a.getNumElements();
        for (int i = 0; i < n; i++) {
            mean += a.get(i);
        }

        return mean / n;
    }

    public static SimpleMatrix eq(SimpleMatrix a, SimpleMatrix b) {
        if (a.numRows() != b.numRows()) throw new IllegalArgumentException("a & b must have an equal number of rows!");
        if (a.numCols() != b.numCols()) throw new IllegalArgumentException("a & b must have an equal number of columns!");

        var eq = new SimpleMatrix(a.numRows(), a.numCols());

        for (int i = 0; i < a.numRows(); i++) {
            for (int j = 0; j < a.numCols(); j++) {
                eq.set(i, j, a.get(i, j) == b.get(i, j) ? 1 : 0);
            }
        }

        return eq;
    }
}
