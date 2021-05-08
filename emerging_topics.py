import argparse
import json
import os.path

from pyspark.sql import functions as F, SparkSession
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA
from pyspark.sql import types as spark_types
from tqdm import tqdm


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

	dataset = spark.read.parquet(dataset_path)

	for index, period in tqdm(enumerate(periods)):

		print("Start detecting emerging topics for period: ", period)
		filtered_dataset = dataset.filter(filter_by_period(period)(F.col('date')))

		num_tweets = filtered_dataset.count()
		print("Number of tweets: ", num_tweets)
		if num_tweets == 0:
			print("No tweets for period:", period)
			continue

		print("Fit tf-idf...")
		cv = CountVectorizer(inputCol="words", outputCol="raw_features", vocabSize=30000, minDF=100, maxDF=10000)
		cvmodel = cv.fit(filtered_dataset)
		featurizedData = cvmodel.transform(filtered_dataset)

		idf = IDF(inputCol="raw_features", outputCol="features")
		idfModel = idf.fit(featurizedData)
		rescaledData = idfModel.transform(featurizedData)
		vocabArray = cvmodel.vocabulary

		# Trains a LDA model.
		print("Fit LDA...")
		lda = LDA(k=5, maxIter=100, featuresCol="features", seed=13)
		model = lda.fit(rescaledData)
		ll = model.logLikelihood(rescaledData)
		print("The lower bound on the log likelihood of the entire corpus: " + str(ll))

		transformed = model.transform(rescaledData)
		transformed.select("id", "topicDistribution").write.json(os.path.join(saving_path, str(index) + "_period"), mode="overwrite")

		# Describe topics.
		print("Extract topics, saving to ", os.path.join(saving_path, str(index) + "_period.json"))
		word_numbers = 10
		topics = model.describeTopics(maxTermsPerTopic=word_numbers).collect()

		topics_to_save = {}
		for topic in topics:
			term_indices = topic['termIndices']
			term_weights = topic['termWeights']
			results = []
			for term_index, term_weight in zip(term_indices, term_weights):
				term = vocabArray[term_index]
				results.append((term, term_weight))

			topics_to_save["topic" + str(topic['topic'])] = results

		with open(os.path.join(saving_path, str(index) + "_period.json"), "w") as fout:
			json.dump(topics_to_save, fout, indent=4)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="preprocessing_us-pres-elections-2020/Hydrated_US_Tweets_preprocessed_emerging_topics"
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
