package be.guldentops.geert.simple.ml;

import org.ejml.simple.SimpleMatrix;

public final class SimpleMatrixUtilities {

    private SimpleMatrixUtilities() {
    }

    public static SimpleMatrix zeros(int numRows) {
        SimpleMatrix columnVector = new SimpleMatrix(numRows, 1);
        columnVector.zero();

        return columnVector;
    }

    public static SimpleMatrix ones(int numRows) {
        SimpleMatrix columnVector = new SimpleMatrix(numRows, 1);
        columnVector.fill(1.0);

        return columnVector;
    }

    public static double mean(SimpleMatrix a) {
        double mean = 0.0;

        int n = a.getNumElements();
        for (int i = 0; i < n; i++) {
            mean += a.get(i);
        }

        return mean / n;
    }

    public static SimpleMatrix eq(SimpleMatrix a, SimpleMatrix b) {
        if (a.numRows() != b.numRows()) throw new IllegalArgumentException("a & b must have an equal number of rows!");
        if (a.numCols() != b.numCols())
            throw new IllegalArgumentException("a & b must have an equal number of columns!");

        SimpleMatrix eq = new SimpleMatrix(a.numRows(), a.numCols());

        for (int i = 0; i < a.numRows(); i++) {
            for (int j = 0; j < a.numCols(); j++) {
                eq.set(i, j, a.get(i, j) == b.get(i, j) ? 1 : 0);
            }
        }

        return eq;
    }

    public static SimpleMatrix eq(SimpleMatrix a, double b) {
        if (a.numCols() != 1) throw new IllegalArgumentException("a must be a column vector!");

        SimpleMatrix eq = new SimpleMatrix(a.numRows(), 1);

        for (int i = 0; i < a.numRows(); i++) {
            eq.set(i, 0, a.get(i, 0) == b ? 1 : 0);
        }

        return eq;
    }

    /**
     * Returns the index of the highest value in the row vector.
     *
     * @param a - the row vector
     * @return the index of the highest value
     */
    public static int maxIndex(SimpleMatrix a) {
        if (a.numRows() != 1) throw new IllegalArgumentException("a must be a row vector!");

        double max = 0.0;
        int maxIndex = -1;
        for (int i = 0; i < a.numCols(); i++) {
            double column = a.get(0, i);
            if (column > max) {
                max = column;
                maxIndex = i;
            }
        }
        
        if (maxIndex == -1) throw new IllegalStateException("Could not find a max index!");

        return maxIndex;
    }
}
