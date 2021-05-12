import glob
import json

import pandas as pd
from tqdm import tqdm

topic_files = glob.glob("data/emerging_topics/period-*.json")

data = {"ID": [], "period": [], "topic_distribution": []}

for topic_file in topic_files:
    period = topic_file.split(".")[0].split("-")[-1]

    with open(topic_file) as f:
        for line in tqdm(f):
            line_data = json.loads(line)
            data["ID"].append(line_data["id"])
            data["period"].append(int(period))
            data["topic_distribution"].append(line_data["topicDistribution"]["values"])
        
    df = pd.DataFrame.from_dict(data)
    df.to_pickle(f"data/emerging_topics/period-{period}.pkl")
