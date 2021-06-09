# Locality_Sensitive_Hashing
This project implements locality sensitive hashing using Spark and MongoDB. 

Locality Sensitive Hashing is a technique to get candidate pair items (i.e items that are similar in some context) in less time complexity. The project work attached here is part of my Assignment work for the course ** Big Data Analytics** at IIIT-Delhi.

The dataset used here (csv file) contains: In each row two questions are given along with their id's. Each row has a duplicate field which states questions given are candidate pair or not (ground truth). Duplicate status = 1 means they are candidate pairs else no.

Pyspark is used to fetch the data from MongoDb database. The implementation of algorithm is done with Spark RDD.

To know more details about the implementation and algorithm kindly read the Report.
