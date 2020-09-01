package be.guldentops.geert.simple.ml.logistic.regression.regularized;

import be.guldentops.geert.simple.ml.Dimensions;
import be.guldentops.geert.simple.ml.MatrixLoader;
import be.guldentops.geert.simple.ml.logistic.regression.regularized.RegularizedLogisticRegression.Hyperparameters;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.eq;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.mean;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.ones;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.zeros;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;
import static org.ejml.EjmlUnitTests.assertShape;

class RegularizedLogisticRegressionTest {

    private RegularizedLogisticRegression algorithm;

    @Nested
    class WhiteBoxTests {

        @BeforeEach
        void setUp() {
            SimpleMatrix trainingSet = new MatrixLoader().load("training-sets/microchip_quality.txt", new Dimensions(118, 3));

            algorithm = new RegularizedLogisticRegression(new Hyperparameters(0.01, 10_000, 1));
            algorithm.learn(trainingSet);
        }

        @Test
        void extractPolynomialsFeatures() {
            SimpleMatrix features = algorithm.features();

            assertShape(features.getMatrix(), 118, 27);

            // Sanity check: only assert first and last row.
            assertThat(features.get(0, 0)).isCloseTo(0.051267, offset(0.000001));
            assertThat(features.get(0, 1)).isCloseTo(0.69956, offset(0.000001));
            assertThat(features.get(0, 2)).isCloseTo(0.0026283, offset(0.000001));
            assertThat(features.get(0, 3)).isCloseTo(0.035864, offset(0.000001));
            assertThat(features.get(117, 0)).isCloseTo(0.63265, offset(0.000001));
            assertThat(features.get(117, 1)).isCloseTo(-0.030612, offset(0.000001));
            assertThat(features.get(117, 2)).isCloseTo(0.40025, offset(0.00001));
            assertThat(features.get(117, 3)).isCloseTo(-0.019367, offset(0.000001));
        }

        @Test
        void extractLabels() {
            SimpleMatrix labels = algorithm.labels();

            assertShape(labels.getMatrix(), 118, 1);

            // Sanity check: only assert first and last row.
            assertThat(labels.get(0, 0)).isEqualTo(1.0);
            assertThat(labels.get(117, 0)).isEqualTo(0);
        }

        @Test
        void costFunctionZeroInitialTheta() {
            SimpleMatrix initialTheta = zeros(algorithm.features().numCols() + 1);
            SimpleMatrix gradient = algorithm.costFunction(algorithm.features(), algorithm.labels(), initialTheta, 1);

            assertShape(gradient.getMatrix(), 28, 1);

            assertThat(gradient.get(0, 0)).isEqualTo(0.0085, offset(0.0001));
            assertThat(gradient.get(1, 0)).isEqualTo(0.0188, offset(0.0001));
            assertThat(gradient.get(2, 0)).isEqualTo(0.0001, offset(0.0001));
            assertThat(gradient.get(3, 0)).isEqualTo(0.0503, offset(0.0001));
            assertThat(gradient.get(4, 0)).isEqualTo(0.0115, offset(0.0001));
        }

        @Test
        void costFunctionNonZeroInitialTheta() {
            SimpleMatrix initialTheta = ones(algorithm.features().numCols() + 1);
            SimpleMatrix gradient = algorithm.costFunction(algorithm.features(), algorithm.labels(), initialTheta, 10);

            assertShape(gradient.getMatrix(), 28, 1);

            assertThat(gradient.get(0, 0)).isEqualTo(0.3460, offset(0.0001));
            assertThat(gradient.get(1, 0)).isEqualTo(0.1614, offset(0.0001));
            assertThat(gradient.get(2, 0)).isEqualTo(0.1948, offset(0.0001));
            assertThat(gradient.get(3, 0)).isEqualTo(0.2269, offset(0.0001));
            assertThat(gradient.get(4, 0)).isEqualTo(0.0922, offset(0.0001));
        }
    }

    @Nested
    class BlackBoxTest {

        @BeforeEach
        void setUp() {
            SimpleMatrix trainingSet = new MatrixLoader().load("training-sets/microchip_quality.txt", new Dimensions(118, 3));

            algorithm = new RegularizedLogisticRegression(new Hyperparameters(0.01, 10_000, 0.1));
            algorithm.learn(trainingSet);
        }

        @Test
        void calculateTrainingAccuracy() {
            SimpleMatrix prediction = algorithm.predictMany(algorithm.features());

            assertShape(prediction.getMatrix(), 118, 1);
            assertThat(mean(eq(prediction, algorithm.labels())) * 100).isEqualTo(86.45, offset(0.01));
        }
    }
}
