import tweepy
import time
import pickle
import os
import re
import itertools
import pandas as pd

tens = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
ones = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
tens_dict = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
              'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
ones_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
             'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
teens_dict = {'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
              'seventeen': 17, 'eighteen': 18, 'nineteen': 19}
def number_strs():
    cartesian = list(itertools.product(tens, ones))
    cartesian.extend([[t] for t in teens])
    return cartesian

def list_in_str(l, s):
    return all(elem in s for elem in l)

def find_number(text):
    result = re.search('I(.*)years old', text)
    if result == None:
        return -1
    tokens = result.group(1).split(' ')
    numbers = [int(t) for t in tokens if t.isnumeric()]
    if len(numbers)==1 and numbers[0] > 10 and numbers[0]<110:
        return numbers[0]
    else:
        return -1
    strs = number_strs()
    for l in strs:
        if list_in_str(l, result.group(1)):
            print(l)
            if len(l)==2:
                return tens_dict[l[0]]+ones_dict[l[1]]
            elif len(l)==1 and l[0] in tens:
                return tens_dict[l[0]]
            elif len(l)==1 and l[0] in ones:
                return ones_dict[l[0]]
            elif len(l)==1 and l[0] in teens:
                return teens_dict[l[0]]
    return -1    

auth = tweepy.OAuthHandler("ye96m2gBpTiiUeI4xketoyRzv",
                           "rZZXot74QF83Yw6ZvAhxEtmw0zNERhq8iODvduw4c0FiRBhElH")
auth.set_access_token("744629802427105280-cuTvt0digJkX7mqogTEfvSnduztwsG7",
                      "rl5MJ63PmPtIfXNfke6hpzvLiqSwdMeRtaZs7tZ1rFPY8")

api = tweepy.API(auth)

total_tweets = []

if not os.path.exists('pretrained-models/user-age/tweet_userID.pkl'):
    for page in tweepy.Cursor(api.search, q='i am i\'m years old -filter:retweets',
                               tweet_mode='extended', count=100).pages():
        texts = [t._json['full_text'] for t in page if not t._json['retweeted']]
        users = [t._json['user']['id'] for t in page if not t._json['retweeted']]  

        total_tweets.extend(list(zip(texts, users)))

        total_tweets = list(set(total_tweets))
        print(len(total_tweets), 'tweets collected')
    pickle.dump(total_tweets, open('pretrained-models/user-age/tweet_userID.pkl', 'wb'))
else:
    total_tweets = pickle.load(open('pretrained-models/user-age/tweet_userID.pkl', 'rb'))

filtered_tweets = []
filtered_users = []
filtered_ages = []

for i in range(len(total_tweets)):
    text = total_tweets[i][0]
    user = total_tweets[i][1]
    age = find_number(text)
    if age > -1:
        filtered_tweets.append(text)
        filtered_users.append(user)
        filtered_ages.append(age)

users_ages = list(zip(filtered_users, filtered_ages))
all_tweets = []
all_ages = []
final_df = None

if not os.path.exists('data/user-age-dataset.csv'):
    i = 0
    for x in users_ages:
        print(round(100*i/len(users_ages), 2), '% done', sep='')
        n_pages = 0
        try:
            for page in tweepy.Cursor(api.user_timeline, user_id=x[0],
                                    tweet_mode='extended', count=100,
                                    include_rts=False).pages():
                texts = [t._json['full_text'] for t in page if not t._json['retweeted']]
                texts = [t.replace(';', '') for t in texts]
                ages = [x[1]]*len(texts)
                all_tweets.extend(texts)
                all_ages.extend(ages)
                n_pages += 1
                if n_pages==3:
                    break
        except:
            time.sleep(900)
        i += 1
    final_df = pd.DataFrame(list(zip(all_tweets, all_ages)), columns=['text', 'age'])
    final_df.to_csv('data/user-age-dataset.csv', index=False)

        
        

