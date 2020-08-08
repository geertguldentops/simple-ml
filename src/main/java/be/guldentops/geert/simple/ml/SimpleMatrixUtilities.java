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
}
