package be.guldentops.geert.simple.ml.logistic.regression;

import org.ejml.simple.SimpleMatrix;

public interface LogisticRegression {

    void learn(SimpleMatrix trainingSet);

    double predictOne(SimpleMatrix newData);

    SimpleMatrix predictMany(SimpleMatrix newData);
}
