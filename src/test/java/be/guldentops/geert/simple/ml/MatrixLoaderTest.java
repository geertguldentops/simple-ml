package be.guldentops.geert.simple.ml;

import org.junit.jupiter.api.Test;

import java.util.List;

import static be.guldentops.geert.simple.ml.MatrixTestUtilities.assertRowVector;
import static org.assertj.core.api.Assertions.assertThat;
import static org.ejml.EjmlUnitTests.assertShape;

class MatrixLoaderTest {

    private final MatrixLoader matrixLoader = new MatrixLoader();

    @Test
    void loadsTrainingSetFoodTruckProfitsPerCity() {
        var trainingSet = matrixLoader.load("training-sets/food-truck-profits-per-city.txt", new Dimensions(97, 2));

        assertThat(trainingSet).isNotNull();
        assertShape(trainingSet.getMatrix(), 97, 2);

        // Sanity check: only assert first and last row.
        assertRowVector(trainingSet.rows(0, 1), List.of(6.1101, 17.592));
        assertRowVector(trainingSet.rows(96, 97), List.of(5.4369, 0.61705));
    }

    @Test
    void loadsTrainingSetHousingPrices() {
        var trainingSet = matrixLoader.load("training-sets/housing_prices.txt", new Dimensions(47, 3));

        assertThat(trainingSet).isNotNull();
        assertShape(trainingSet.getMatrix(), 47, 3);

        // Sanity check: only assert first and last row.
        assertRowVector(trainingSet.rows(0, 1), List.of(2104.0, 3.0, 399_900.0));
        assertRowVector(trainingSet.rows(46, 47), List.of(1203.0, 3.0, 239_500.0));
    }
}
