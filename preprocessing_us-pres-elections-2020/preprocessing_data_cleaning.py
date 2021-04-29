from pyspark.sql import functions as F, SparkSession
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql import types as spark_types
import argparse
import re
from nltk.tokenize import TreebankWordTokenizer

@F.udf(returnType=spark_types.StringType())
def clean_tweet(text):
	cleaned_text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).lower()
	cleaned_text = TreebankWordTokenizer().tokenize(cleaned_text)
	# stemmed_text = [
	# 	stemmer.stem(word) for word in cleaned_text
	# 	if word not in stop_words and len(word) > 3]
	if len(cleaned_text) < 4:
		return None
	else:
		return " ".join(cleaned_text)


def preprocess_dataset(dataset_path, output_path):

	spark = SparkSession.builder.master("local[*]"). \
		appName("preprocess"). \
		getOrCreate()

	cleaned_dataset = spark.read.csv(dataset_path, header=True)\
		.filter(F.col('text').isNotNull())\
		.select("ID", 'date', clean_tweet(F.col('text')).alias("text"))

	filtered_dataset = cleaned_dataset.filter(F.col("text").isNotNull())

	filtered_dataset.coalesce(1).write.csv(output_path, header=True, mode="overwrite")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="tweets-us20oct-9nov.csv"
	)

	parser.add_argument(
		"--output_path",
		"-o",
		help="path to save the preprocessed dataset",
		default="cleaned_tweets-us20oct-9nov"
	)
	args = parser.parse_args()

	preprocess_dataset(args.dataset_path, args.output_path)
