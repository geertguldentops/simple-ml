package be.guldentops.geert.simple.ml.logistic.regression;

import be.guldentops.geert.simple.ml.Dimensions;
import be.guldentops.geert.simple.ml.Hyperparameters;
import be.guldentops.geert.simple.ml.MatrixLoader;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.zeros;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;
import static org.ejml.EjmlUnitTests.assertShape;

class LogisticRegressionTest {

    private MultivariateLogisticRegression algorithm;

    @BeforeEach
    void setUp() {
        var trainingSet = new MatrixLoader().load("training-sets/student_admission.txt", new Dimensions(100, 3));

        algorithm = new MultivariateLogisticRegression(new Hyperparameters(0.01, 400));
        algorithm.learn(trainingSet);
    }

    @Nested
    class WhiteBoxTests {

        @Test
        void extractFeatures() {
            var features = algorithm.features();

            assertShape(features.getMatrix(), 100, 2);

            // Sanity check: only assert first and last row.
            assertThat(features.get(0, 0)).isEqualTo(34.62365962451697);
            assertThat(features.get(0, 1)).isEqualTo(78.0246928153624);
            assertThat(features.get(99, 0)).isEqualTo(74.77589300092767);
            assertThat(features.get(99, 1)).isEqualTo(89.52981289513276);
        }

        @Test
        void extractLabels() {
            var labels = algorithm.labels();

            assertShape(labels.getMatrix(), 100, 1);

            // Sanity check: only assert first and last row.
            assertThat(labels.get(0, 0)).isEqualTo(0);
            assertThat(labels.get(99, 0)).isEqualTo(1);
        }

        @Test
        void costFunctionZeroInitialTheta() {
            var initialTheta = zeros(algorithm.features().numCols() + 1);
            var gradient = algorithm.costFunction(algorithm.features(), algorithm.labels(), initialTheta);

            assertShape(gradient.getMatrix(), 3, 1);

            assertThat(gradient.get(0, 0)).isEqualTo(-0.1000, offset(0.0001));
            assertThat(gradient.get(1, 0)).isEqualTo(-12.0092, offset(0.0001));
            assertThat(gradient.get(2, 0)).isEqualTo(-11.2628, offset(0.0001));
        }

        @Test
        void costFunctionNonZeroInitialTheta() {
            var initialTheta = new SimpleMatrix(new double[][]{{-24}, {0.2}, {0.2}});
            var gradient = algorithm.costFunction(algorithm.features(), algorithm.labels(), initialTheta);

            assertShape(gradient.getMatrix(), 3, 1);
            assertThat(gradient.get(0, 0)).isEqualTo(0.043, offset(0.001));
            assertThat(gradient.get(1, 0)).isEqualTo(2.566, offset(0.001));
            assertThat(gradient.get(2, 0)).isEqualTo(2.647, offset(0.001));
        }

        @Test
        void model() {
            var model = algorithm.model();

            assertShape(model.getMatrix(), 3, 1);
            assertThat(model.get(0, 0)).isEqualTo(-25.161, offset(0.001));
            assertThat(model.get(1, 0)).isEqualTo(0.206, offset(0.001));
            assertThat(model.get(2, 0)).isEqualTo(0.201, offset(0.001));
        }
    }

    @Nested
    class BlackBoxTest {

        @Test
        void predictsAdmissionOfStudentWithExamScores45And85() {
            var newData = new SimpleMatrix(new double[][]{{45, 85}});

            var predictAdmission = algorithm.predictOne(newData);

            assertThat(predictAdmission).isEqualTo(1);
        }

        @Test
        void calculateTrainingAccuracy() {
            var predictions = algorithm.predictMany(algorithm.features());

            assertShape(predictions.getMatrix(), 100, 1);
            assertThat(mean(eq(predictions, algorithm.labels())) * 100).isEqualTo(89.0, offset(0.1));
        }
    }

    private static double mean(SimpleMatrix a) {
        var mean = 0.0;

        var n = a.getNumElements();
        for (int i = 0; i < n; i++) {
            mean += a.get(i);
        }

        return mean / n;
    }

    private static SimpleMatrix eq(SimpleMatrix a, SimpleMatrix b) {
        if (a.numRows() != b.numRows()) throw new IllegalArgumentException("a & b must have equal # rows!");
        if (a.numCols() != b.numCols()) throw new IllegalArgumentException("a & b must have equal # columns!");

        var eq = new SimpleMatrix(a.numRows(), b.numCols());

        for (int i = 0; i < a.numRows(); i++) {
            for (int j = 0; j < a.numCols(); j++) {
                eq.set(i, j, a.get(i, j) == b.get(i, j) ? 1 : 0);
            }
        }

        return eq;
    }
}
