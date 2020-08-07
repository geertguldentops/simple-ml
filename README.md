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
 
    This makes it a perfect training set for univariate linear regression


## How do I get set up? ##

### Compiling the code ###

Maven (3) is used to compile the code and run the tests: 

* You can run maven manually using `mvn clean install` from the command line.
* You can run the cleanInstall.sh script, e.g.: 
    * `> chmod +x cleanInstall.sh`
    * `> ./cleanInstall.sh` 
* Run the project's only pom.xml file using your favourite IDE! (IntelliJ IDEA, Eclipse, Netbeans, etc.)
