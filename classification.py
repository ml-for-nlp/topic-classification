""" Topic Classification -- SVM tutorial

Usage:
  classification.py --C=<n> --kernel=<kn> [--degree=<n>] [--show-urls] [--real-queries]
  classification.py (-h | --help)
  classification.py --version

Options:
  -h --help           Show this screen.
  --version           Show version.
  --C=<n>                 Value of C parameter.
  --kernel=<kn>            Type of kernel.
  --degree=<n>            Degree of kernel.
  --show-urls         Optional. Show URLs corresponding to support vectors.
  --real-queries      Optional. Test system on search queries.

"""


import os
from sys import argv
from docopt import docopt
from numpy import concatenate
from sklearn.svm import SVC
from utils import *


args = docopt(__doc__, version='Topic Classification 1.0')
#print(args)
C       = int(args["--C"])
kernel  = args["--kernel"]
degree  = int(args["--degree"]) if args["--degree"] else 3
showurls = True if args["--show-urls"] else False
realtest = True if args["--real-queries"] else False

#csv sources
files = [os.path.join('./data/', f) for f in os.listdir('./data/') \
	if os.path.isfile(os.path.join('./data/', f)) \
	and os.path.join('./data/', f)[-4:] == ".csv"]

#fetch data in csv and structure in a topic:vector dict
topic_dict = get_data(files)

#get user-selected topics
topics = list(topic_dict.keys())
print('Available topics:\n{}\n'.format(format_topics(topics)))
t1 = t2 = ''
t1 = get_topic(t1, topics, 1)
t2 = get_topic(t2, topics, 2)
print()

#get user-selected size of training sets
train1_size = get_train_size(t1, topic_dict[t1])
train2_size = get_train_size(t2, topic_dict[t2])
print()

#build numpy arrays and lists of urls
t1_train, t1_test, t1_train_urls, t1_test_urls = \
	make_arrays(topic_dict[t1], train1_size)
t2_train, t2_test, t2_train_urls, t2_test_urls = \
	make_arrays(topic_dict[t2], train2_size)
train_urls = t1_train_urls + t2_train_urls

test1_size = len(t1_test)
test2_size = len(t2_test)

#print on topic-wise size of sets
print('Topic 1: Train size: {} | Test size: {}\n' \
	.format(train1_size, test1_size) + \
	'Topic 2: Train size: {} | Test size: {}\n' \
	.format(train2_size, test2_size))
	
#prepare train/test sets
x_train = concatenate([t1_train, t2_train])
x_test  = concatenate([t1_test,  t2_test])
y_train = make_labels(train1_size, train2_size)
y_test  = make_labels(test1_size,  test2_size)

#setup SVM setup and print output
print('SVC output:')
clf    = SVC(C = C, verbose = True, kernel = kernel, degree = degree) #prints data
model  = clf.fit(x_train, y_train)
score  = clf.score(x_test, y_test)
y_pred = clf.predict(x_test)

print('\n') #needed because SVC prints output in a weird way
print('SVC Model:')
print(model)
print()

print('Score: {}\n'.format(score))

if showurls: #very annoying output
	print('Training urls:')
	print('\n'.join([train_urls[s] for s in clf.support_]))
	print()

#make confusion matrix
make_confmat(y_pred, y_test, t1, t2) #prints some data

#test on real queries
if len(argv) > 3 and realtest:
	print('Test on real queries:\n' + \
		'(Topic 1: {} | Topic 2: {})\n'.format(t1, t2))
	queries = get_queries('data/queryvectors.txt') #dict
	for key, value in queries.items():
		print(key, clf.predict(value.reshape(1, -1)))
