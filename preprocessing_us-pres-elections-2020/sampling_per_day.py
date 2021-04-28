from pyspark.sql import SparkSession
import os
from tqdm import tqdm


def sampling():
    dataset_path = "../us-pres-elections-2020"

    days_per_month_dir = {"09": 30, "10": 31, "11": 30}

    spark = SparkSession.builder.master("local[*]"). \
        appName("sampling"). \
        getOrCreate()

    for month in os.listdir(dataset_path):
        if month == '.DS_Store' or month == '.git':
            continue
        month_path = os.path.join(dataset_path, month)
        days_per_month = days_per_month_dir[month.split('-')[1]]
        if not os.path.exists(month):
            os.mkdir(month)
        print(month)
        for day in tqdm(range(1, days_per_month + 1)):
            if day < 10:
                day_str = "0" + str(day)
            else:
                day_str = str(day)

            day_dataset = spark.read.text(month_path + '/us-presidential-tweet-id-' + month + '-' + day_str + '-*.txt')
            day_dataset.persist()
            size = day_dataset.count()
            sampled_day_dataset = day_dataset.sample(fraction=21000 / size, seed=4)
            day_dataset.unpersist()
            twitter_ids = sampled_day_dataset.collect()
            twitter_ids = [row['value'] for row in twitter_ids]
            with open(month + "/" + day_str + '.txt', 'w') as fout:
                fout.write("\n".join(twitter_ids))


if __name__ == '__main__':
    sampling()









