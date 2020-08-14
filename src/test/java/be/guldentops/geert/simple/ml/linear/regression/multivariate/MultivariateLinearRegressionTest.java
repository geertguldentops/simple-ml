package be.guldentops.geert.simple.ml.linear.regression.multivariate;

import be.guldentops.geert.simple.ml.Dimensions;
import be.guldentops.geert.simple.ml.Hyperparameters;
import be.guldentops.geert.simple.ml.MatrixLoader;
import be.guldentops.geert.simple.ml.linear.regression.LinearRegression;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

public class MultivariateLinearRegressionTest {

    private LinearRegression algorithm;

    @BeforeEach
    void setUp() {
        var trainingSet = new MatrixLoader().load("training-sets/housing_prices.txt", new Dimensions(47, 3));

        algorithm = new MultivariateLinearRegression(new Hyperparameters(0.01, 400));
        algorithm.learn(trainingSet);
    }

    @Nested
    class BlackBoxTest {

        @Test
        void predictsPriceOf1650SquareFeet3BedroomHouse() {
            var newData = new SimpleMatrix(new double[][]{{1_650, 3}});

            var predictedPrice = algorithm.predict(newData);

            assertThat(predictedPrice).isCloseTo(289_314.620338, offset(0.000001));
        }
    }
}
