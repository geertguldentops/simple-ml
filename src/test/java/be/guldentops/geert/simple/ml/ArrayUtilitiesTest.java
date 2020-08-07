package be.guldentops.geert.simple.ml;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class ArrayUtilitiesTest {

	@Test
	void createArrayOfAllZerosOfVariousSizes() {
		assertThat(ArrayUtilities.zeros(1)).hasSize(1).containsExactly(0.0);
		assertThat(ArrayUtilities.zeros(2)).hasSize(2).containsExactly(0.0, 0.0);
		assertThat(ArrayUtilities.zeros(3)).hasSize(3).containsExactly(0.0, 0.0, 0.0);
	}

	@Test
	void createArrayOfAllOnesOfVariousSizes() {
		assertThat(ArrayUtilities.ones(1)).hasSize(1).containsExactly(1.0);
		assertThat(ArrayUtilities.ones(2)).hasSize(2).containsExactly(1.0, 1.0);
		assertThat(ArrayUtilities.ones(3)).hasSize(3).containsExactly(1.0, 1.0, 1.0);
	}
}
