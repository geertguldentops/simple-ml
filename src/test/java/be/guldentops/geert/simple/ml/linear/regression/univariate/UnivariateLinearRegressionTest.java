package be.guldentops.geert.simple.ml.linear.regression.univariate;

import be.guldentops.geert.simple.ml.Dimensions;
import be.guldentops.geert.simple.ml.Hyperparameters;
import be.guldentops.geert.simple.ml.MatrixLoader;
import org.ejml.data.Matrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.ejml.EjmlUnitTests.assertShape;

class UnivariateLinearRegressionTest {

	private UnivariateLinearRegression algorithm;
	private Matrix trainingSet;

	@BeforeEach
	void setUp() {
		this.algorithm = new UnivariateLinearRegression(new Hyperparameters(0.01, 1500));
		this.trainingSet = new MatrixLoader().load("training-sets/food-truck-profits-per-city.txt", new Dimensions(97, 2));
	}

	@Nested
	class WhiteBoxTests {

		@Test
		void addsBiasToFeatures() {
			algorithm.learn(trainingSet);

			var features = algorithm.features();

			assertShape(features.getMatrix(), 97, 2);

			// Sanity check: only assert first and last row.
			assertThat(features.get(0, 0)).isEqualTo(1.0);
			assertThat(features.get(0, 1)).isEqualTo(6.1101);
			assertThat(features.get(96, 0)).isEqualTo(1.0);
			assertThat(features.get(96, 1)).isEqualTo(5.4369);
		}

		@Test
		void extractLabels() {
			algorithm.learn(trainingSet);

			var labels = algorithm.labels();

			assertShape(labels.getMatrix(), 97, 1);

			// Sanity check: only assert first and last row.
			assertThat(labels.get(0, 0)).isEqualTo(17.592);
			assertThat(labels.get(96, 0)).isEqualTo(0.61705);
		}

		@Test
		void learnsModelThatFitsTrainingSet() {
			algorithm.learn(trainingSet);

			var model = algorithm.model();

			assertShape(model.getMatrix(), 2, 1);

			// Note: There is actually no way to know these values up front, otherwise it would not be machine learning!
			assertThat(model.get(0, 0)).isEqualTo(-3.63029143940436);
			assertThat(model.get(1, 0)).isEqualTo(1.166362350335582);
		}
	}

}
