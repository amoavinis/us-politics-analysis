import argparse
from pyspark.sql import functions as F, SparkSession
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA
from pyspark.sql import types as spark_types
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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
		return period['start'] < (date[0], date[1]) < period['end']
	return F.udf(f, spark_types.BooleanType())


def explore_emerging_topics(dataset_path):

	spark = SparkSession.builder.master("local[*]"). \
		appName("emerging_topics"). \
		getOrCreate()

	dataset = spark.read.csv(dataset_path, header=True).filter(F.col("date").isNotNull())

	dataset = dataset.select("ID", transform_date("date").alias('date'), "text")
	dataset.show()
	print(dataset.count())
	period = {'start': (10, 21), 'end': (10, 22)}
	print(dataset.filter(filter_by_period(period)(F.col('date'))).count())
	exit()
	preprocessed_dataset = dataset.select("ID", "date", text_preprocessing("text").alias("words"))

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
	print("The topics described by their top-weighted terms:")
	word_numbers = 10
	topics = model.describeTopics(maxTermsPerTopic=word_numbers).collect()

	for topic in topics:
		print("Topic" + str(topic['topic']) + ":")
		term_indices = topic['termIndices']
		term_weights = topic['termWeights']
		results = []
		for term_index, term_weight in zip(term_indices, term_weights):
			term = vocabArray[term_index]
			results.append((term, term_weight))
		for result in results:
			print(result)
		print('\n')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="preprocessing_us-pres-elections-2020/cleaned_tweets-us20oct-9nov"
	)
	args = parser.parse_args()
	explore_emerging_topics(args.dataset_path)
