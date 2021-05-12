import glob
import json

import pandas as pd
from tqdm import tqdm


topic_files = glob.glob("data/emerging_topics/period-*.json")

# keep all periods
all_data = {"ID": [], "period": [], "topic_distribution": []}

for topic_file in topic_files:
    period = topic_file.split(".")[0].split("-")[-1]

    # keep each period
    data = {"ID": [], "period": [], "topic_distribution": []}

    with open(topic_file) as f:
        for line in tqdm(f):
            line_data = json.loads(line)
            data["ID"].append(line_data["id"])
            data["period"].append(int(period))
            data["topic_distribution"].append(line_data["topicDistribution"]["values"])

    # save each period 
    df = pd.DataFrame.from_dict(data)
    df.to_pickle(f"data/emerging_topics/period-{period}.pkl")

    for key in all_data:
        all_data[key] += data[key]

# save all period
df_all = pd.DataFrame.from_dict(all_data)
df_all.to_pickle(f"data/emerging_topics/period-all.pkl")
