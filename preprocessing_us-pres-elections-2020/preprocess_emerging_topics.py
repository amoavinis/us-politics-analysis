import argparse

from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import preprocessor
from nltk.tokenize import TreebankWordTokenizer

preprocessor.set_options(
	preprocessor.OPT.URL,
	preprocessor.OPT.MENTION,
	preprocessor.OPT.EMOJI,
	preprocessor.OPT.SMILEY,
	preprocessor.OPT.NUMBER,
	preprocessor.OPT.ESCAPE_CHAR,
)

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
stemmer = PorterStemmer()
month_dir = {'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11}


@F.udf(returnType=spark_types.ArrayType(spark_types.StringType()))
def text_preprocessing(text):
	cleaned_text = preprocessor.clean(text)
	stemmed_words = [stemmer.stem(word) for word in TreebankWordTokenizer().tokenize(cleaned_text)
		if word not in stop_words and word not in punctuations and len(word) > 3]

	if len(stemmed_words) < 4:
		return None
	else:
		return stemmed_words


@F.udf(returnType=spark_types.ArrayType(spark_types.IntegerType()))
def transform_date(date_text):
	month = date_text.split()[1]
	month = month_dir[month]
	day = int(date_text.split()[2])

	return [month, day]


def preprocess_emerging_topics(dataset_path, output_path):

	spark = SparkSession.builder.master("local[*]"). \
		appName("emerging_topics"). \
		getOrCreate()

	dataset = spark.read.json(dataset_path).filter(F.col("date").isNotNull())

	print("Initial dataset: ", dataset.count())

	dataset = dataset.distinct().select("ID", transform_date("date").alias('date'), "text")

	preprocessed_dataset = dataset.select("ID", "date", text_preprocessing("text").alias("words")) \
		.filter(F.col("words").isNotNull())

	preprocessed_dataset.persist()
	print("Preprocessed dataset: ", preprocessed_dataset.count())
	preprocessed_dataset.coalesce(1).write.parquet(output_path, mode="overwrite")
	preprocessed_dataset.unpersist()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="Hydrated US Tweets"
	)
	parser.add_argument(
		"--output_path",
		"-o",
		help="output path to save preprocessed dataset",
		default="Hydrated_US_Tweets_preprocessed_emerging_topics"
	)
	args = parser.parse_args()

	preprocess_emerging_topics(args.dataset_path, args.output_path)