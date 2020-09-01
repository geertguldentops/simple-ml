package be.guldentops.geert.simple.ml.logistic.regression.onevsall;

import be.guldentops.geert.simple.ml.Dimensions;
import be.guldentops.geert.simple.ml.MatrixLoader;
import be.guldentops.geert.simple.ml.logistic.regression.onevsall.OneVsAllLogisticRegression.Hyperparameters;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.eq;
import static be.guldentops.geert.simple.ml.SimpleMatrixUtilities.mean;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;
import static org.ejml.EjmlUnitTests.assertShape;

class OneVsAllLogisticRegressionTest {

    @Nested
    class CostFunctionTests {

        private OneVsAllLogisticRegression oneVsAllLogisticRegression;

        @BeforeEach
        public void setUp() {
            oneVsAllLogisticRegression = new OneVsAllLogisticRegression(null);
        }

        @Test
        void costFunctionNonZeroInitialThetaWithLambda3() {
            SimpleMatrix theta = new SimpleMatrix(new double[][]{
                    {-2},
                    {-1},
                    {1},
                    {2},
            });
            SimpleMatrix features = new SimpleMatrix(new double[][]{
                    {0.1, 0.6, 1.1},
                    {0.2, 0.7, 1.2},
                    {0.3, 0.8, 1.3},
                    {0.4, 0.9, 1.4},
                    {0.5, 1, 1.5},
            });
            SimpleMatrix labels = new SimpleMatrix(new double[][]{
                    {1},
                    {0},
                    {1},
                    {0},
                    {1},
            });

            SimpleMatrix gradient = oneVsAllLogisticRegression.costFunction(features, labels, theta, 3.0);

            assertShape(gradient.getMatrix(), 4, 1);

            assertThat(gradient.get(0, 0)).isEqualTo(0.146561367924898, offset(0.0000001));
            assertThat(gradient.get(1, 0)).isEqualTo(-0.5485584118531603, offset(0.00000001));
            assertThat(gradient.get(2, 0)).isEqualTo(0.7247222721092885, offset(0.0000000001));
            assertThat(gradient.get(3, 0)).isEqualTo(1.3980029560717375, offset(0.0000000001));
        }

        @Test
        void costFunctionNonZeroInitialThetaWithLambda01() {
            SimpleMatrix theta = new SimpleMatrix(new double[][]{
                    {-2},
                    {-1},
                    {1},
                    {2},
            });
            SimpleMatrix features = new SimpleMatrix(new double[][]{
                    {0.1, 0.6, 1.1},
                    {0.2, 0.7, 1.2},
                    {0.3, 0.8, 1.3},
                    {0.4, 0.9, 1.4},
                    {0.5, 1, 1.5},
            });
            SimpleMatrix labels = new SimpleMatrix(new double[][]{
                    {1},
                    {0},
                    {1},
                    {0},
                    {1},
            });

            SimpleMatrix gradient = new OneVsAllLogisticRegression(null).costFunction(features, labels, theta, 0.1);

            assertShape(gradient.getMatrix(), 4, 1);

            assertThat(gradient.get(0, 0)).isEqualTo(0.146561, offset(0.000001));
            assertThat(gradient.get(1, 0)).isEqualTo(0.031442, offset(0.000001));
            assertThat(gradient.get(2, 0)).isEqualTo(0.144722, offset(0.000001));
            assertThat(gradient.get(3, 0)).isEqualTo(0.238003, offset(0.000001));
        }
    }

    @Nested
    class BlackBoxTests {

        private OneVsAllLogisticRegression algorithm;

        @BeforeEach
        void setUp() {
            SimpleMatrix trainingSet = new MatrixLoader().load("training-sets/handwritten_digits.txt", new Dimensions(5_000, 401));

            algorithm = new OneVsAllLogisticRegression(new Hyperparameters(0.01, 400, 0.1, 10));
            algorithm.learn(trainingSet);
        }

        @Test
        void calculateTrainingAccuracy() {
            SimpleMatrix predictions = algorithm.predictMany(algorithm.features());

            assertShape(predictions.getMatrix(), 5_000, 1);
            assertThat(mean(eq(predictions, algorithm.labels())) * 100).isEqualTo(94.9, offset(0.1));
        }
    }
}
