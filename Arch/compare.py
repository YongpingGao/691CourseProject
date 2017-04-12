
from __future__ import print_function

 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
 

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
def compare_sentences( sentences_a, sentences_b ):

    # Init our vectorizer
    vect = TfidfVectorizer( min_df = 1 )

    # Create our tfidf
    tfidf = vect.fit_transform( [ sentences_a, sentences_b ] )

    # Get an array of results
    results = ( tfidf * tfidf.T ).A

    # Return percentage float
    # return float( '%.4f' % ( results[0][1] * 100 ) )
    return results[0][1]

print(compare_sentences("from sklearn.feature_extraction.text import TfidfTransformer", "from sklearn.feature_extraction.text import HashingVectorizer"))
