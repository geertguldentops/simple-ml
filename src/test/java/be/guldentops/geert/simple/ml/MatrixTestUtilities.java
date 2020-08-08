package be.guldentops.geert.simple.ml;

import org.ejml.simple.SimpleMatrix;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.ejml.EjmlUnitTests.assertShape;

public final class MatrixTestUtilities {

    private MatrixTestUtilities() {
    }

    public static void assertRowVector(SimpleMatrix rowVector, List<Double> expectedValues) {
        assertShape(rowVector.getMatrix(), 1, expectedValues.size());

        for (int i = 0; i < expectedValues.size(); i++) {
            assertThat(rowVector.get(0, i)).isEqualTo(expectedValues.get(i));
        }
    }

    public static void assertColumnVector(SimpleMatrix columnVector, List<Double> expectedValues) {
        assertShape(columnVector.getMatrix(), expectedValues.size(), 1);

        for (int i = 0; i < expectedValues.size(); i++) {
            assertThat(columnVector.get(i, 0)).isEqualTo(expectedValues.get(i));
        }
    }

}
