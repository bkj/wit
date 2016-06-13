import pandas as pd
import numpy as np
from faker import Factory

CHOICES = [
    'last_name',
    'user_name',
    'street_address',
    'street_name',
    'credit_card_number',
    'address', 
    'date', 
    'iso8601',
    'ipv4', 
    'ipv6', 
    'free_email', 
    'sha256',
    'url',
    'year',
    'zipcode',
    'language_code',
    'job',
    'file_name',
    'mac_address',
    'ssn',
    'safari',
    'firefox',
    'phone_number'
]


class FakeData:
    def __init__(self, choices=CHOICES):
        self.fake    = Factory.create()
        self.choices = choices
    
    def datapoint(self):
        label = np.random.choice(self.choices)
        return {
            "label" : label,
            "value" : getattr(self.fake, label)()
        }
    
    def dataset(self, size=1000):
        return [self.datapoint() for i in xrange(size)]
