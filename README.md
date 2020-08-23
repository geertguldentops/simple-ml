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
    | 34.62365962451697             | 78.0246928153624      | 0 (was not admitted)      |
    | 30.28671076822607             | 43.89499752400101     | 0 (was not admitted)      |
    | 60.18259938620976             | 86.30855209546826     | 1 (was admitted)          |
    | ...                           | ...                   | ...                       |
 
    This makes it a perfect training set for (multivariate) logistic regression (without requiring regularisation)

## How do I get set up? ##

### Compiling the code ###

Maven (3) is used to compile the code and run the tests: 

* You can run maven manually using `mvn clean install` from the command line.
* You can run the cleanInstall.sh script, e.g.: 
    * `> chmod +x cleanInstall.sh`
    * `> ./cleanInstall.sh` 
* Run the project's only pom.xml file using your favourite IDE! (IntelliJ IDEA, Eclipse, Netbeans, etc.)
