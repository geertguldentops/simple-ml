package be.guldentops.geert.simple.ml;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import java.util.List;

import static be.guldentops.geert.simple.ml.MatrixTestUtilities.assertRowVector;
import static org.assertj.core.api.Assertions.assertThat;

class TrainingSetLoaderTest {

	private final TrainingSetLoader trainingSetLoader = new TrainingSetLoader();

	@Test
	void loadsTrainingSetFoodTruckProfitsPerCity() {
		var trainingSet = trainingSetLoader.load("training-sets/food-truck-profits-per-city.txt", new Dimensions(97, 2));

		assertThat(trainingSet).isNotNull();
		assertThat(trainingSet.getNumRows()).isEqualTo(97);
		assertThat(trainingSet.getNumCols()).isEqualTo(2);

		// Sanity check: only assert first and last row.
		var simpleMatrix = SimpleMatrix.wrap(trainingSet);
		assertRowVector(simpleMatrix.rows(0, 1), List.of(6.1101, 17.592));
		assertRowVector(simpleMatrix.rows(96, 97), List.of(5.4369, 0.61705));
	}

}
