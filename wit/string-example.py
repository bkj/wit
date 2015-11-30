# --
# WIT :: String similarity Example

from wit import *

num_features = 1000
max_len      = 50

# Generate fake dataset
print 'WIT :: Generating data'
f    = FakeData()
data = f.dataframe(size = 5000)

# Format for keras training
print 'WIT :: Formatting data'
formatter   = KerasFormatter(num_features, max_len)
train, levs = formatter.format(data, ['obj'], 'hash')

# Compile and train classifier
print 'WIT :: Compiling classifier'
classifier = StringClassifier(train, levs)
classifier.fit(batch_size = 300)

# Create test dataset
print 'WIT :: Creating test dataset'
testdata = f.dataframe(size = 5000)
test, _  = formatter.format(testdata, ['obj'], 'hash')

# Make prediction on test dataset and check accuracy
print 'WIT :: Predicting on test dataset'
preds = classifier.predict(test['x'])

print 'WIT :: Prediction confusion matrix'
pred_class = np.array(levs)[preds.argmax(1)]
act_class  = np.array(levs)[test['y'].argmax(1)]
print(pd.crosstab(pred_class, act_class))

# Some examples:
assert(classifier.classify_string('http://www.gophronesis.com') == 'url')
assert(classifier.classify_string('ben@gophronesis.com')        == 'free_email')
assert(classifier.classify_string('2000-01-01')                 == 'date')
assert(classifier.classify_string('203-802-9283')               == 'phone_number')
assert(classifier.classify_string('2038329382')                 == 'phone_number')
assert(classifier.classify_string('241-34-1983')                == 'ssn')
assert(classifier.classify_string('10.1.70.234')                == 'ipv4')
assert(classifier.classify_string('andy46')                     == 'user_name')
assert(classifier.classify_string('41177432883439283')          == 'credit_card_number')
assert(classifier.classify_string('2001')                       == 'year')
