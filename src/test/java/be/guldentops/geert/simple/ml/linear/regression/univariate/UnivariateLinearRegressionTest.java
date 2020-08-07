package be.guldentops.geert.simple.ml.linear.regression.univariate;

import be.guldentops.geert.simple.ml.Dimensions;
import be.guldentops.geert.simple.ml.MatrixLoader;
import org.ejml.data.Matrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.ejml.EjmlUnitTests.assertShape;

class UnivariateLinearRegressionTest {

	@Nested
	class StepStoneTests {

		private UnivariateLinearRegression algorithm;
		private Matrix trainingSet;

		@BeforeEach
		void setUp() {
			this.algorithm = new UnivariateLinearRegression();
			this.trainingSet = new MatrixLoader().load("training-sets/food-truck-profits-per-city.txt", new Dimensions(97, 2));
		}

		@Test
		public void addsBiasToFeature() {
			algorithm.learn(trainingSet);

			var X = algorithm.X();

			assertShape(X.getMatrix(), 97, 2);

			// Sanity check: only assert first and last row.
			assertThat(X.get(0, 0)).isEqualTo(1.0);
			assertThat(X.get(0, 1)).isEqualTo(6.1101);
			assertThat(X.get(96, 0)).isEqualTo(1.0);
			assertThat(X.get(96, 1)).isEqualTo(5.4369);
		}
	}

}
