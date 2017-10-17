# spark-languagedetector 

This is a language detection algorithm using n-grams of bytes to use with the big data framework Spark.  

For a given dataset of traininng data it computes each gram's probability of occurrence in a language. For the prediction, it
scans a text with the given gram sizes and multiplies the probabilities together. 

Currently the estimator expects an already clean training dataset, i.e. a wikipedia article dump where you remove the the punktuation (except "'") and squash whitespaces etc.

It is not yet properly tested and should currently not be used in production, however I'm pretty confident it works ;)


Feel free to use it, fork it, tell your friends about it. If you have questions or suggestions open an issue. 
Pull requests are always welcome.

