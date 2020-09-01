package be.guldentops.geert.simple.ml;

public final class Dimensions {

    private final int rows;
    private final int columns;

    public Dimensions(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }
}
