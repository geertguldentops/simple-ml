package be.guldentops.geert.simple.ml.logistic.regression.multivariate;

import be.guldentops.geert.simple.ml.Dimensions;
import be.guldentops.geert.simple.ml.MatrixLoader;
import be.guldentops.geert.simple.ml.logistic.regression.multivariate.MultivariateLogisticRegression.Hyperparameters;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.eq;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.mean;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.zeros;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;
import static org.ejml.EjmlUnitTests.assertShape;

class MultivariateLogisticRegressionTest {

    private MultivariateLogisticRegression algorithm;

    @BeforeEach
    void setUp() {
        SimpleMatrix trainingSet = new MatrixLoader().load("training-sets/student_admission.txt", new Dimensions(100, 3));

        algorithm = new MultivariateLogisticRegression(new Hyperparameters(0.01, 10_000));
        algorithm.learn(trainingSet);
    }

    @Nested
    class WhiteBoxTests {

        @Test
        void extractFeatures() {
            SimpleMatrix features = algorithm.features();

            assertShape(features.getMatrix(), 100, 2);

            // Sanity check: only assert first and last row.
            assertThat(features.get(0, 0)).isEqualTo(34.62365962451697);
            assertThat(features.get(0, 1)).isEqualTo(78.0246928153624);
            assertThat(features.get(99, 0)).isEqualTo(74.77589300092767);
            assertThat(features.get(99, 1)).isEqualTo(89.52981289513276);
        }

        @Test
        void extractLabels() {
            SimpleMatrix labels = algorithm.labels();

            assertShape(labels.getMatrix(), 100, 1);

            // Sanity check: only assert first and last row.
            assertThat(labels.get(0, 0)).isEqualTo(0);
            assertThat(labels.get(99, 0)).isEqualTo(1);
        }

        @Test
        void costFunctionZeroInitialTheta() {
            SimpleMatrix initialTheta = zeros(algorithm.features().numCols() + 1);
            SimpleMatrix gradient = algorithm.costFunction(algorithm.features(), algorithm.labels(), initialTheta);

            assertShape(gradient.getMatrix(), 3, 1);

            assertThat(gradient.get(0, 0)).isEqualTo(-0.1000, offset(0.0001));
            assertThat(gradient.get(1, 0)).isEqualTo(-0.279819456602434, offset(0.00000000001));
            assertThat(gradient.get(2, 0)).isEqualTo(-0.249728062093054, offset(0.00000000001));
        }

        @Test
        void costFunctionNonZeroInitialTheta() {
            SimpleMatrix initialTheta = new SimpleMatrix(new double[][]{{-24}, {0.2}, {0.2}});
            SimpleMatrix gradient = algorithm.costFunction(algorithm.features(), algorithm.labels(), initialTheta);

            assertShape(gradient.getMatrix(), 3, 1);
            assertThat(gradient.get(0, 0)).isEqualTo(-0.5999999999607762, offset(0.00000000001));
            assertThat(gradient.get(1, 0)).isEqualTo(-0.27981945659506, offset(0.000000001));
            assertThat(gradient.get(2, 0)).isEqualTo(-0.2497280620855351, offset(0.00000000001));
        }

        @Test
        void model() {
            SimpleMatrix model = algorithm.model();

            assertShape(model.getMatrix(), 3, 1);
            assertThat(model.get(0, 0)).isEqualTo(1.2677701988489238, offset(0.00000000001));
            assertThat(model.get(1, 0)).isEqualTo(3.0555058686572587, offset(0.00000000001));
            assertThat(model.get(2, 0)).isEqualTo(2.8189190133012114, offset(0.00000000001));
        }
    }

    @Nested
    class BlackBoxTest {

        @Test
        void predictsAdmissionOfStudentWithExamScores45And85() {
            SimpleMatrix newData = new SimpleMatrix(new double[][]{{45, 85}});

            double predictAdmission = algorithm.predictOne(newData);

            assertThat(predictAdmission).isEqualTo(1.0);
        }

        @Test
        void calculateTrainingAccuracy() {
            SimpleMatrix predictions = algorithm.predictMany(algorithm.features());

            assertShape(predictions.getMatrix(), 100, 1);
            assertThat(mean(eq(predictions, algorithm.labels())) * 100).isEqualTo(89.0, offset(0.1));
        }
    }
}
