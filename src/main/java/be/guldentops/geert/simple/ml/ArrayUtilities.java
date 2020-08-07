package be.guldentops.geert.simple.ml;

import java.util.Arrays;

public final class ArrayUtilities {

    private ArrayUtilities() {
    }

    public static double[] ones(int size) {
        var ones = new double[size];
        Arrays.fill(ones, 1.0);

        return ones;
    }

    public static double[] zeros(int size) {
        var zeros = new double[size];
        Arrays.fill(zeros, 0.0);

        return zeros;
    }
}
