from wit import *

# --
# String similarity Example

num_features = 1000
max_len      = 50

# Generate fake dataset
f    = FakeData()
data = f.dataframe(size = 5000)

# Format for keras training
formatter        = KerasFormatter(num_features, max_len)
train, val, levs = formatter.format_with_val(data, ['obj'], 'hash')

# Compile and train classifier
classifier = StringClassifier(train, val, levs)
classifier.fit()

# Create test dataset
testdata = f.dataframe(size = 5000)
test     = formatter.format(testdata, ['obj'], 'hash')

# Make prediction on test dataset and check accuracy
preds = classifier.predict(test['x'])

pred_class = np.array(levs)[preds.argmax(1)]
act_class  = np.array(levs)[test['y'].argmax(1)]
pd.crosstab(pred_class, act_class)


# Examples
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
