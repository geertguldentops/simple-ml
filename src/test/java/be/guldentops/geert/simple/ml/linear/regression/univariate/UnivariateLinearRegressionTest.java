package be.guldentops.geert.simple.ml.linear.regression.univariate;

import be.guldentops.geert.simple.ml.Dimensions;
import be.guldentops.geert.simple.ml.Hyperparameters;
import be.guldentops.geert.simple.ml.MatrixLoader;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;
import static org.ejml.EjmlUnitTests.assertShape;

class UnivariateLinearRegressionTest {

	private UnivariateLinearRegression algorithm;

	@BeforeEach
	void setUp() {
		var trainingSet = new MatrixLoader().load("training-sets/food-truck-profits-per-city.txt", new Dimensions(97, 2));

		this.algorithm = new UnivariateLinearRegression(new Hyperparameters(0.01, 1500));
		this.algorithm.learn(trainingSet);
	}

	@Nested
	class WhiteBoxTests {

		@Test
		void addsBiasToFeatures() {
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
			var labels = algorithm.labels();

			assertShape(labels.getMatrix(), 97, 1);

			// Sanity check: only assert first and last row.
			assertThat(labels.get(0, 0)).isEqualTo(17.592);
			assertThat(labels.get(96, 0)).isEqualTo(0.61705);
		}

		@Test
		void learnsModelThatFitsTrainingSet() {
			var model = algorithm.model();

			assertShape(model.getMatrix(), 2, 1);

			// Note: There is actually no way to know these values up front, otherwise it would not be machine learning!
			assertThat(model.get(0, 0)).isEqualTo(-3.63029143940436);
			assertThat(model.get(1, 0)).isEqualTo(1.166362350335582);
		}
	}

	@Nested
	class BlackBoxTests {

		@Test
		void predictAccuratelyForAPopulationOf35k() {
			var newData = new SimpleMatrix(new double[][] { { 3.5 } });

			double _35kPrediction = algorithm.predict(newData);

			// Note: There is actually no way to know this value up front, otherwise it would not be machine learning!
			assertThat(_35kPrediction).isCloseTo(0.4520, offset(0.0001));
		}

		@Test
		void predictAccuratelyForAPopulationOf70k() {
			var newData = new SimpleMatrix(new double[][] { { 7.0 } });

			double _70kPrediction = algorithm.predict(newData);

			// Note: There is actually no way to know this value up front, otherwise it would not be machine learning!
			assertThat(_70kPrediction).isEqualTo(4.5342, offset(0.0001));
		}
	}
}
