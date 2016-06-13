'''
    WIT :: String similarity example
'''

import pandas as pd
import numpy as np
from pprint import pprint

import sys
sys.path.insert(0, '..')
from wit.wit import StringClassifier
from wit.datasets import FakeData

num_features = 1000 # Characters
max_len      = 75  # Characters

# Generate fake dataset
print >> sys.stderr, '\nstring_example.py :: Generating fake data, eg:'
faker      = FakeData()
train_data = faker.dataset(size=50000)
test_data  = faker.dataset(size=1000)
pprint(train_data[0:10])

# Compile and train classifier
print '\nstring_example.py :: Training model'
classifier = StringClassifier(num_features=num_features, max_len=max_len)
_ = classifier.fit(train_data, **{
    "batch_size" : 256, 
    "nb_epoch" : 15, 
    "verbose" : True
})

# Make prediction on test dataset and check accuracy
print >> sys.stderr, '\nstring_example.py :: Prediction confusion matrix'
actual_classes = np.array([x['label'] for x in test_data])
predicted_classes = classifier.predict([x['value'] for x in test_data])
print pd.crosstab(predicted_classes, actual_classes)
print np.mean(predicted_classes == actual_classes)

# Show performance on some examples
print >> sys.stderr, '\nstring_example.py :: Trying some examples'
examples = [
    'http://www.gophronesis.com',
    'ben@gophronesis.com',
    '2000-01-01',
    '1-203-802-9283',
    '12038329382',
    '241-34-1983',
    '10.1.60.201',
    'randy46',
    '41177432883439283',
    '2001'
]
predicted_classes = classifier.predict(examples)
for i,ex in enumerate(examples):
    print 'input: %s' % ex
    print 'predicted: %s\n' % predicted_classes[i]
