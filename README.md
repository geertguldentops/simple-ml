# README #

## What is this repository for? ##

Some very simple machine learning algorithms written in vanilla Java and a single linear algebra library [EJML](http://ejml.org/wiki/index.php?title=Main_Page)

### Training sets

A brief explanation of the training sets:

1.  food-truck-profits-per-city.txt is a training set about the profits of a food truck

    It contains 1 feature and a label:

    | City population   | Profit        |
    | -------------     |:-------------:|
    | 6.1101            | 17.592        |
    | 5.5277            | 9.1302        |
    | 8.5186            | 13.662        |
    | ...               | ...           |
 
    This makes it a perfect training set for univariate linear regression

2.  housing_prices.txt is a training set about the housing prices in a city

    It contains 2 features and a label:

    | Size of the house (in square feet)    | Number of bedrooms    | price of the house in $   |
    | -------------                         |:---------------------:|:-------------------------:|
    | 2104                                  | 3                     | 399900                    |
    | 1600                                  | 3                     | 329900                    |
    | 2400                                  | 3                     | 369000                    |
    | ...                                   | ...                   | ...                       |
 
    This makes it a perfect training set for multivariate linear regression

3.  student_admission.txt is a training set about student study results and whether they were admitted to university or not

    It contains 2 features and a label:

    | Penultimate exam result       | Final exam result     | Admitted to university    |
    | -------------                 |:---------------------:|:-------------------------:|
    | 34.62365962451697             | 78.0246928153624      |   0 (not admitted)        |
    | 30.28671076822607             | 43.89499752400101     |   0 (not admitted)        |
    | 60.18259938620976             | 86.30855209546826     |   1 (was admitted)        |
    | ...                           | ...                   | ...                       |
 
    This makes it a perfect training set for (multivariate) logistic regression (without requiring regularization)
    
4.  microchip_quality.txt is a training set about microchips from a fabrication plant passing quality assurance or not.

    It contains 2 features and a label:

    | QA test 1 result      | QA test 2 result     | Accepted or rejected   |
    | -------------         |:--------------------:|:----------------------:|
    | 0.051267              |   0.69956            | 1 (accepted)           |
    | -0.092742             |   0.68494            | 1 (accepted)           |
    | 0.18376               |   0.93348            | 0 (accepted)           |
    | ...                   | ...                  | ...                    |
    
    This makes it a perfect training set for (multivariate) logistic regression (with regularization)
    
5.  handwritten_digits.txt is a training set containing handwritten digits, it is a subset of data of the 
    MNIST handwritten digit dataset (http://yann.lecun.com/exdb/mnist/) as a training set
    
    Each training example is a 20 pixel by 20 pixel grayscale image of the digit. 
    Each pixel is represented by a floating point number indicating the grayscale intensity at that location. 
    The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. 
    Each of these training examples becomes a single row in our data matrix X. 
    This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image.
    
    The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. 
    To make things more compatible with Octave/MATLAB indexing, where there is no zero index, the digit zero is mapped to the value ten. 
    Therefore, a “0” digit is labeled as “10”, while the digits “1” to “9” are labeled as “1” to “9” in their natural order.
 
    This makes it a perfect training set for one-vs-all classification problem

## How do I get set up? ##

### Compiling the code ###

Maven (3) is used to compile the code and run the tests: 

* You can run maven manually using `mvn clean install` from the command line.
* You can run the cleanInstall.sh script, e.g.: 
    * `> chmod +x cleanInstall.sh`
    * `> ./cleanInstall.sh` 
* Run the project's only pom.xml file using your favourite IDE! (IntelliJ IDEA, Eclipse, Netbeans, etc.)
