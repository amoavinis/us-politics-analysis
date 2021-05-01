import json
import sys
import os
import pandas
import numpy


numpy.set_printoptions(suppress=True)

args = sys.argv

input_dir = args[1]
output_file = args[2]

ids = []
tweets = []
dates = []

for filename in os.listdir(input_dir):
    the_file = input_dir + filename
    with open(the_file) as file:
        for line in file:
            data = json.loads(line)

            # If it's a retweet get the original status to avoid information loss
            if data.get('retweeted_status'):
                tweets.append(data['retweeted_status']['full_text'])
            else:
                tweets.append(data['full_text'])
            
            ids.append(int(data['id']))
            dates.append(data["created_at"])

all_data = list(zip(ids, dates, tweets))
all_data = sorted(all_data, key=lambda l:l[0])

df = pandas.DataFrame(all_data, columns=['ID', 'date', 'text'])
df.to_json(output_file, orient='records')


# How to run:
# $ python jsonl_to_csv.py outputs/ tweets-us-{DATE}.json
