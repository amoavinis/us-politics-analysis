import argparse
import json
import os.path

from pyspark.sql import functions as F, SparkSession
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA
from pyspark.sql import types as spark_types
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
month_dir = {'Sep': 9, 'Oct': 10, 'Nov': 11}


@F.udf(returnType=spark_types.ArrayType(spark_types.StringType()))
def text_preprocessing(text):
	stemmed_text = [stemmer.stem(word) for word in text.split(" ")
		if word not in stop_words and len(word) > 3]

	return stemmed_text


@F.udf(returnType=spark_types.ArrayType(spark_types.IntegerType()))
def transform_date(date_text):
	month = date_text.split()[1]
	month = month_dir[month]
	day = int(date_text.split()[2])

	return [month, day]


def filter_by_period(period):
	def f(date):
		return (period['start'][0], period['start'][1]) <= (date[0], date[1]) \
			   <= (period['end'][0], period['end'][1])
	return F.udf(f, spark_types.BooleanType())


def explore_emerging_topics(dataset_path, periods_path, saving_path):

	with open(periods_path, "r") as periods_json:
		periods = json.load(periods_json)

	if not os.path.exists(saving_path):
		os.mkdir(saving_path)

	spark = SparkSession.builder.master("local[*]"). \
		appName("emerging_topics"). \
		getOrCreate()

	dataset = spark.read.csv(dataset_path, header=True).filter(F.col("date").isNotNull())

	dataset = dataset.select("ID", transform_date("date").alias('date'), "text")

	for index, period in tqdm(enumerate(periods)):
		filtered_dataset = dataset.filter(filter_by_period(period)(F.col('date')))

		preprocessed_dataset = filtered_dataset.select("ID", "date", text_preprocessing("text").alias("words"))

		cv = CountVectorizer(inputCol="words", outputCol="raw_features", vocabSize=30000, minDF=100, maxDF=10000)
		cvmodel = cv.fit(preprocessed_dataset)
		featurizedData = cvmodel.transform(preprocessed_dataset)

		idf = IDF(inputCol="raw_features", outputCol="features")
		idfModel = idf.fit(featurizedData)
		rescaledData = idfModel.transform(featurizedData)
		vocabArray = cvmodel.vocabulary

		# rescaledData.select("words", "features").show(truncate=False)

		# Trains a LDA model.
		lda = LDA(k=5, maxIter=100, featuresCol="features", seed=13)
		model = lda.fit(rescaledData)

		# ll = model.logLikelihood(rescaledData)
		# lp = model.logPerplexity(rescaledData)
		# print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
		# print("The upper bound on perplexity: " + str(lp))

		# Describe topics.
		# print("The topics described by their top-weighted terms:")
		word_numbers = 10
		topics = model.describeTopics(maxTermsPerTopic=word_numbers).collect()

		topics_to_save = {}
		for topic in topics:
			# print("Topic" + str(topic['topic']) + ":")
			term_indices = topic['termIndices']
			term_weights = topic['termWeights']
			results = []
			for term_index, term_weight in zip(term_indices, term_weights):
				term = vocabArray[term_index]
				results.append((term, term_weight))

			topics_to_save["topic" + str(topic['topic'])] = results
			# for result in results:
			# 	print(result)
			# print('\n')

		with open(os.path.join(saving_path, str(index) + "_period.json"), "w") as fout:
			json.dump(topics_to_save, fout, indent=4)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="preprocessing_us-pres-elections-2020/cleaned_tweets-us20oct-9nov"
	)
	parser.add_argument(
		"--periods_path",
		"-p",
		help="periods json file path",
		default="periods.json"
	)
	parser.add_argument(
		"--saving_path",
		"-s",
		help="saving file path ",
		default="emerging_topics_results"
	)
	args = parser.parse_args()

	explore_emerging_topics(
		args.dataset_path,
		args.periods_path,
		args.saving_path)
