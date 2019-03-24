# SVMs for topic classification

This is an implementation of an SVM classifier for topic classification, given a set of documents expressed as vectors.

### The data

In your data/ directory, you will find some csv files created by the [PeARS](http://pearsearch.org/) search engine. These files contain representations of web documents as 400-dimensional vectors, sorted by topics (one file per topic).

You will first need to decompress all files in that directory. Do:

    cd data
    for f in *bz2; do bzip2 -d $f; done
    cd ..

### Running the SVM

The SVM implementation used here comes from the [scikit-learn toolkit](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC). It lets you build a classifier for the data using a range of different kernels and various parameter values.

Run the classifier with e.g.

    python3 classification.py --C=100 --kernel=linear

Instead of *linear*, you could also use *poly* for a polynomial kernel or *rbf* for an RBF Gaussian kernel. 100 is the C hyperparameter, which you can also change. When using *poly*, you can indicate the degree of the polynomial with e.g.

    python3 classification.py --C=100 --kernel=poly --degree=5

The program will ask you to choose two topics out of a list (remember that SVMs are binary classifiers). For instance:

    ['string theory', 'harry potter', 'star wars', 'black panther film', 'black panthers', 'opera']
    Enter topic 1: harry potter
    Enter topic 2: star wars

You can then choose how many training instances you want to use for each class. E.g.:

    star wars has 787 documents. How many do you want for training? 400
    star trek has 415 documents. How many do you want for training? 100

We then get the output of the SVM, the score over the test data, and the URL corresponding to the support vectors for the classification. Two confusion matrices will also be printed as .png in your directory, showing the errors made by the system (one version shows error frequencies, and the other percentages of errors).

In addition, you can output the URLs corresponding to the support vectors for the classification using the --show-urls flag:

    python3 classification.py --C=100 --kernel=linear --show-urls


### Inspecting support vectors

Run the classifier with different kernels and notice the difference in number of support vectors selected by the classifier (that is the *nSV* value in the output). What do you conclude?

Open the web pages for a few support vectors. What do you notice?


### Understanding the effect of class distribution

Try and play with the number of documents you use for training in each class. What do you notice? Is there a correlation between how balanced the training data is and how close the topics are? For instance, compare unbalanced training data in the case of *star wars* vs *star trek* and *star wars* vs *clunch*.


### Analysing search engine queries

Now, for the ultimate test... add the --real-queries flag when you run the classifier:

    python3 classification.py --C=100 --kernel=linear --real-queries

This will output a set of queries, and which of the two chosen classes they were classified into. We are checking here whether, for the queries that are actually related to one of the two classes, the assignment is correct.

Why do you think things work on such small `documents'? (Queries are only a few words long...)

The vectors for documents and queries are normalised (see utils.py). Do you understand why?


