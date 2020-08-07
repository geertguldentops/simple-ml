package be.guldentops.geert.simple.ml;

import org.ejml.data.Matrix;
import org.ejml.ops.MatrixIO;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Paths;

public class TrainingSetLoader {

	public Matrix load(String fileName, Dimensions dimensions) {
		var absoluteFilePath = createAbsoluteFilePath(fileName);
		var rows = dimensions.rows();
		var columns = dimensions.columns();

		try {
			return MatrixIO.loadCSV(absoluteFilePath, rows, columns);
		} catch (IOException e) {
			throw new RuntimeException(String.format("Could not read file %s into a matrix %dx%d", fileName, rows, columns), e);
		}
	}

	private String createAbsoluteFilePath(String fileName) {
		try {
			return Paths.get(ClassLoader.getSystemResource(fileName).toURI()).toAbsolutePath().toString();
		} catch (URISyntaxException e) {
			throw new RuntimeException(String.format("Could resolve absolute file path of %s", fileName), e);
		}
	}
}
