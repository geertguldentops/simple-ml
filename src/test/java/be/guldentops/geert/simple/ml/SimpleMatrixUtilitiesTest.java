package be.guldentops.geert.simple.ml;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.eq;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.mean;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.ones;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.zeros;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.ejml.EjmlUnitTests.assertEquals;
import static org.ejml.EjmlUnitTests.assertShape;

class SimpleMatrixUtilitiesTest {

    @Test
    void createSimpleMatrixOfAllZerosOfVariousSizes() {
        assertShape(zeros(1).getMatrix(), 1, 1);
        assertThat(zeros(1).isVector()).isTrue();
        assertThat(zeros(1).get(0, 0)).isEqualTo(0.0);

        assertShape(zeros(2).getMatrix(), 2, 1);
        assertThat(zeros(2).isVector()).isTrue();
        assertThat(zeros(2).get(0, 0)).isEqualTo(0.0);
        assertThat(zeros(2).get(1, 0)).isEqualTo(0.0);

        assertShape(zeros(3).getMatrix(), 3, 1);
        assertThat(zeros(3).isVector()).isTrue();
        assertThat(zeros(3).get(0, 0)).isEqualTo(0.0);
        assertThat(zeros(3).get(1, 0)).isEqualTo(0.0);
        assertThat(zeros(3).get(2, 0)).isEqualTo(0.0);
    }

    @Test
    void createSimpleMatrixOfAllOnesOfVariousSizes() {
        assertShape(ones(1).getMatrix(), 1, 1);
        assertThat(ones(1).isVector()).isTrue();
        assertThat(ones(1).get(0, 0)).isEqualTo(1.0);

        assertShape(ones(2).getMatrix(), 2, 1);
        assertThat(ones(2).isVector()).isTrue();
        assertThat(ones(2).get(0, 0)).isEqualTo(1.0);
        assertThat(ones(2).get(1, 0)).isEqualTo(1.0);

        assertShape(ones(3).getMatrix(), 3, 1);
        assertThat(ones(3).isVector()).isTrue();
        assertThat(ones(3).get(0, 0)).isEqualTo(1.0);
        assertThat(ones(3).get(1, 0)).isEqualTo(1.0);
        assertThat(ones(3).get(2, 0)).isEqualTo(1.0);
    }

    @Test
    void calculatesMeanOfAMatrix() {
        assertThat(mean(new SimpleMatrix(new double[][]{{1, 1}, {1, 1}}))).isEqualTo(1);
        assertThat(mean(new SimpleMatrix(new double[][]{{1, 2}, {3, 4}}))).isEqualTo(2.5);
    }

    @Test
    void equalityMatrixWithEqualMatrices() {
        var eqMatrix = eq(
                new SimpleMatrix(new double[][]{{1, 2}, {3, 4}}),
                new SimpleMatrix(new double[][]{{1, 2}, {3, 4}})
        );

        assertEquals(eqMatrix.getMatrix(), new SimpleMatrix(new double[][]{{1, 1}, {1, 1}}).getMatrix());
    }

    @Test
    void equalityMatrixWithCompletelyDifferentMatrices() {
        var eqMatrix = eq(
                new SimpleMatrix(new double[][]{{1, 2}, {3, 4}}),
                new SimpleMatrix(new double[][]{{5, 6}, {7, 8}})
        );

        assertEquals(eqMatrix.getMatrix(), new SimpleMatrix(new double[][]{{0, 0}, {0, 0}}).getMatrix());
    }

    @Test
    void equalityMatrixWithPartiallyDifferentMatrices() {
        var eqMatrix = eq(
                new SimpleMatrix(new double[][]{{1, 2}, {3, 4}}),
                new SimpleMatrix(new double[][]{{1, 6}, {3, 8}})
        );

        assertEquals(eqMatrix.getMatrix(), new SimpleMatrix(new double[][]{{1, 0}, {1, 0}}).getMatrix());
    }

    @Test
    void equalityMatrixWithMatricesDifferentRows() {
        assertThatThrownBy(() ->
                eq(
                        new SimpleMatrix(new double[][]{{1, 2}}),
                        new SimpleMatrix(new double[][]{{1, 6}, {3, 8}})
                )
        ).isInstanceOf(IllegalArgumentException.class)
                .hasMessage("a & b must have an equal number of rows!");
    }

    @Test
    void equalityMatrixWithMatricesDifferentColumns() {
        assertThatThrownBy(() ->
                eq(
                        new SimpleMatrix(new double[][]{{1}, {3}}),
                        new SimpleMatrix(new double[][]{{1, 6}, {3, 8}})
                )
        ).isInstanceOf(IllegalArgumentException.class)
                .hasMessage("a & b must have an equal number of columns!");
    }

    @Test
    void equalityMatrixWithEqual() {
        var eqMatrix = eq(
                new SimpleMatrix(new double[][]{{2}, {2}, {2}}),
                2
        );

        assertEquals(eqMatrix.getMatrix(), new SimpleMatrix(new double[][]{{1}, {1}, {1}}).getMatrix());
    }

    @Test
    void equalityMatrixWithCompletelyDifferent() {
        var eqMatrix = eq(
                new SimpleMatrix(new double[][]{{1}, {3}, {4}}),
                2
        );

        assertEquals(eqMatrix.getMatrix(), new SimpleMatrix(new double[][]{{0}, {0}, {0}}).getMatrix());
    }

    @Test
    void equalityMatrixWithPartiallyDifferent() {
        var eqMatrix = eq(
                new SimpleMatrix(new double[][]{{1}, {3}, {4}, {3}}),
                3
        );

        assertEquals(eqMatrix.getMatrix(), new SimpleMatrix(new double[][]{{0}, {1}, {0}, {1}}).getMatrix());
    }

    @Test
    void equalityMatrixWithNotColumnVector() {
        assertThatThrownBy(() ->
                eq(
                        new SimpleMatrix(new double[][]{{1, 2}, {3, 8}}),
                        2
                )
        ).isInstanceOf(IllegalArgumentException.class)
                .hasMessage("a must be a column vector!");
    }
}
