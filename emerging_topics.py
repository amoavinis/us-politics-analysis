from pyspark.sql import functions as F, SparkSession
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql import types as spark_types
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()