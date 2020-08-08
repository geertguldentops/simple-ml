package be.guldentops.geert.simple.ml;

import org.junit.jupiter.api.Test;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.ones;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.zeros;
import static org.assertj.core.api.Assertions.assertThat;
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
}
