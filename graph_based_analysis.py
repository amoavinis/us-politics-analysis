import networkx as nx
import pandas as pd


def get_edge_list():
    
    # get edge list from tweets that are retweets
    df_retweets = pd.read_pickle("data/tweets-us-all-with-retweets.pkl")
    df_retweets = df_retweets.dropna(subset=["retweet_graph_edge"])
    edge_list = df_retweets["retweet_graph_edge"].tolist()
    print(f"\nWe found {len(edge_list)} graph edges.")

    return edge_list


def export_sample_graph(n_nodes):

    # get edges
    edge_list = get_edge_list()

    # build graph from edge list
    G = nx.Graph()
    G.add_edges_from(edge_list[:n_nodes])

    # save for gephi
    nx.write_gexf(G, "data/sample-retweet-graph.gexf")


def get_most_retweeted_users(n):

    # get edges
    edge_list = get_edge_list()

    # build graph from edge list
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # get most retweetes users
    most_retweeted = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:n]
    
    print("\nMost retweeted users (user_id, retweets):")
    for user_id, retweets in most_retweeted:
        print(user_id, ",", retweets)


def get_most_retweeted_tweets(n):

    # read dataset
    df_retweets = pd.read_pickle("data/tweets-us-all-with-retweets.pkl")
    df_retweets = df_retweets.dropna(subset=["retweet_graph_edge"])

    # add original user id column
    df_retweets["original_user_id"] = df_retweets["retweet_graph_edge"].apply(lambda x: x[1])

    # get most retweeted tweets
    most_retweeted = df_retweets["text"].value_counts()[:n]

    # get corresponding ids of the original users
    df_retweets_original = df_retweets[["text", "original_user_id"]].drop_duplicates()

    print("\nMost retweeted tweets (Original tweet", "Number of retweets", "Original user id):")
    for text, retweets in most_retweeted.iteritems():
        user_id = df_retweets_original[df_retweets_original["text"] == text]["original_user_id"].values[0]
        print("\n")
        print("Tweet:", text)
        print("Retweets:", retweets)
        print("User id:", user_id)


if __name__ == "__main__":

    export_sample_graph(n_nodes=10000)

    get_most_retweeted_users(n=40)

    get_most_retweeted_tweets(n=40)


# Most retweeted users (user_id, retweets):
#
# 939091 , 38210
# 30354991 , 12539
# 226222147 , 6220
# 216776631 , 5478
# 1205226529455632385 , 5282
# 32871086 , 4766
# 1640929196 , 4682
# 255812611 , 4666
# 1243560408025198593 , 3998
# 39344374 , 3798
# 292929271 , 3645
# 14247236 , 3623
# 592730371 , 3066
# 803694179079458816 , 2998
# 1073047860260814848 , 2976
# 357606935 , 2913
# 1108472017144201216 , 2831
# 78523300 , 2579
# 2228878592 , 2537
# 236487888 , 2525
# 90480218 , 2379
# 471677441 , 2307
# 18266688 , 2208
# 963790885937995777 , 2188
# 1058520120 , 2107
# 38495835 , 2089
# 259001548 , 2083
# 288277167 , 2048
# 225265639 , 2033
# 27493883 , 2002
# 1043185714437992449 , 1942
# 1212806053907185664 , 1839
# 970207298 , 1732
# 15976705 , 1670
# 15115280 , 1573
# 91882544 , 1559
# 21059255 , 1556
# 216065430 , 1550
# 1917731 , 1514
# 39349894 , 1477


# Twitter ID to @handle conversion
#
# 939091 => @JoeBiden
# 30354991 => @KamalaHarris
# 226222147 => @PeteButtigieg
# 216776631 => @BernieSanders
# 1205226529455632385 => @ProjectLincoln
# 32871086 => @kylegriffin1
# 1640929196 => @mmpadellan
# 255812611 => @donwinslow
# 1243560408025198593 => @MeidasTouch
# 39344374 => @DonaldJTrumpJr
# 292929271 => @charliekirk11
# 14247236 => @funder
# 592730371 => @JackPosobiec
# 803694179079458816 => @VP
# 1073047860260814848 => @CaslerNoel
# 357606935 => @ewarren
# 1108472017144201216 => @TrumpWarRoom
# 78523300 => @RealJamesWoods
# 2228878592 => @AndrewYang
# 236487888 => @WalshFreedom
# 90480218 => @RichardGrenell
# 471677441 => @gtconway3d
# 18266688 => @TomFitton
# 963790885937995777 => @HKrassenstein
# 1058520120 => @SenDuckworth
# 38495835 => @marklevinshow
# 259001548 => @kayleighmcenany
# 288277167 => @atrupar
# 225265639 => @ddale8
# 27493883 => @joncoopertweets
# 1043185714437992449 => @catturd2
# 1212806053907185664 => @TheLeoTerrell
# 970207298 => @SenWarren
# 15976705 => @Amy_Siskind
# 15115280 => @PalmerReport
# 91882544 => @DineshDSouza
# 21059255 => @tedlieu
# 216065430 => @staceyabrams
# 1917731 => @thehill
# 39349894 => @EricTrump



# Most retweeted tweets (Original tweet Number of retweets Original user id):
# 
# 
# Tweet: While I may be the first woman in this office, I will not be the last—because every little girl watching tonight sees that this is a country of possibilities.
# Retweets: 845
# User id: 30354991
# 
# 
# Tweet: Keep the faith, guys. We’re gonna win this.
# Retweets: 578
# User id: 939091
# 
# 
# Tweet: I won't be president until January 20th, but my message today to everyone is this: wear a mask.
# Retweets: 545
# User id: 939091
# 
# 
# Tweet: I’m Joe Biden and I approve this message. https://t.co/TuRZXPE5xK
# Retweets: 538
# User id: 939091
# 
# 
# Tweet: I’m happy to report that Jill and I have tested negative for COVID. Thank you to everyone for your messages of concern. I hope this serves as a reminder: wear a mask, keep social distance, and wash your hands.
# Retweets: 426
# User id: 939091
# 
# 
# Tweet: The fires across the West Coast are just the latest examples of the very real ways our changing climate is changing our communities. Protecting our planet is on the ballot. Vote like your life depends on it—because it does. https://t.co/gKGegXWxQu
# Retweets: 415
# User id: 813286
# 
# 
# Tweet: America is back.
# Retweets: 407
# User id: 939091
# 
# 
# Tweet: One month until Election Day. Let’s do this, America.
# Retweets: 404
# User id: 939091
# 
# 
# Tweet: Count every vote.
# Retweets: 389
# User id: 939091
# 
# 
# Tweet: Mr. President, if you don’t respect our troops, you can’t lead them. https://t.co/hcX9hGgdm5
# Retweets: 364
# User id: 939091
# 
# 
# Tweet: Wearing a mask isn't a political statement — it's a patriotic duty.
# Retweets: 355
# User id: 939091
# 
# 
# Tweet: It’s Election Day. Go vote, America!
# Retweets: 325
# User id: 939091
# 
# 
# Tweet: Donald Trump said he didn't want to tell the truth and create a panic. So he did nothing and created a disaster.
# Retweets: 316
# User id: 939091
# 
# 
# Tweet: https://t.co/TtWm3i4eaq.
# Retweets: 305
# User id: 939091
# 
# 
# Tweet: .@JoeBiden DOES NOT want to defund the police. Please retweet this to help beat back the @realDonaldTrump big lie.
# Retweets: 300
# User id: 24578794
# 
# 
# Tweet: I’ve released 21 years of my tax returns. 
# 
# What are you hiding, @realDonaldTrump? https://t.co/aQs6Hlox0P
# Retweets: 293
# User id: 939091
# 
# 
# Tweet: Wear a mask. Wash your hands. Vote out Donald Trump.
# Retweets: 291
# User id: 939091
# 
# 
# Tweet: We did this—together.
# Retweets: 271
# User id: 30354991
# 
# 
# Tweet: Let me be clear: Wearing a mask is not about making your life less comfortable or taking something away. It’s to give something back to all of us — a normal life.
# Retweets: 269
# User id: 939091
# 
# 
# Tweet: My husband John lived by a code: country first. We are Republicans, yes, but Americans foremost. There's only one candidate in this race who stands up for our values as a nation, and that is @JoeBiden.
# Retweets: 269
# User id: 61263431
# 
# 
# Tweet: My father died of Covid alone in a hospital. I had to say goodbye to him over a phone. Trump got a joyride to sooth his desperate need for attention, while endangering the lives of the Secret Service people in the car with him. To hell with him and all who enable him.
# Retweets: 266
# User id: 933128688
# 
# 
# Tweet: Let me be clear: The voters should pick a President, and that President should select a successor to Justice Ginsburg.
# Retweets: 265
# User id: 939091
# 
# 
# Tweet: Did the President of the United States just instruct a white supremacist group to “stand by”?
# Retweets: 250
# User id: 226222147
# 
# 
# Tweet: Isn’t Pres Trump currently the President so he’s actually touring his America. https://t.co/Aj4nmk6pEj
# Retweets: 250
# User id: 371539066
# 
# 
# Tweet: 25 days. Let’s win this thing.
# Retweets: 250
# User id: 939091
# 
# 
# Tweet: This is exactly the rhetoric that has put me, my family, and other government officials’ lives in danger while we try to save the lives of our fellow Americans. It needs to stop. https://t.co/EWkNQx3Ppx
# Retweets: 243
# User id: 102071743
# 
# 
# Tweet: Listening to scientists is not a bad thing.
# 
# I can’t believe that has to be said.
# Retweets: 234
# User id: 939091
# 
# 
# Tweet: We need to remember: We’re at war with a virus — not with each other.
# Retweets: 230
# User id: 939091
# 
# 
# Tweet: 1 day. Let’s win this thing.
# Retweets: 229
# User id: 939091
# 
# 
# Tweet: It’s not enough to praise our essential workers — we have to protect and pay them.
# Retweets: 226
# User id: 939091
# 
# 
# Tweet: I want to congratulate all those who worked so hard to make this historic day possible. Now, through our continued grassroots organizing, let us create a government that works for ALL and not the few. Let us create a nation built on justice, not greed and bigotry.
# Retweets: 222
# User id: 216776631
# 
# 
# Tweet: We may be opponents — but we are not enemies. 
# 
# We are Americans.
# Retweets: 219
# User id: 939091
# 
# 
# Tweet: Georgia, thank you. Together, we have changed the course of our state for the better. But our work is not done. 
# 
# Join me in supporting @ReverendWarnock and @ossoff so we can keep up the fight and win the U.S. Senate➡️https://t.co/JTyH1UVEtd 
# 
# #LetsGetItDoneAgain #gapol https://t.co/qH5ZfmsgI7
# Retweets: 219
# User id: 216065430
# 
# 
# Tweet: Stay in line, folks.
# Retweets: 214
# User id: 939091
# 
# 
# Tweet: Listen to the scientists.
# Retweets: 212
# User id: 939091
# 
# 
# Tweet: did u know that trump has TWENTY-SIX SEXUAL ASSAULT ALLEGATIONS filed in court?? and SEVEN are against CHILDREN? How did we allow this monster to be our president? These types of records wouldn’t even hold up for a job in the food industry.
# Retweets: 212
# User id: 3105833401
# 
# 
# Tweet: When Trump was saying young people couldn’t get coronavirus, he knew they could.
# 
# When Trump was saying it was the same as the flu, he knew it was deadlier.
# 
# When Trump was purposely downplaying the severity, he knew it passed through the air. 
# 
# He knew.
# Retweets: 207
# User id: 30354991
# 
# 
# Tweet: I’m running as a Democrat, but I will be an American president. Whether you voted for me or against me, I will represent you.
# Retweets: 207
# User id: 939091
# 
# 
# Tweet: You deserve a president who tells you the truth.
# Retweets: 206
# User id: 939091
# 
# 
# Tweet: I thought we were supposed to keep politics out of sports... or is that just when we stand up for the basic human rights of black lives??? 🤔 https://t.co/k2nRUR2Tu8
# Retweets: 205
# User id: 254353879


# Twitter ID to @handle conversion
#
# 226222147 => @PeteButtigieg
# 813286 => @BarackObama
# 61263431 => @cindymccain
# 24578794 => @Scaramucci
# 30354991 => @KamalaHarris
# 933128688 => @curtisisbooger
# 939091 => @JoeBiden
# 216065430 => @staceyabrams
# 216776631 => @BernieSanders
# 3105833401 => @BrizeidaRuiz
# 371539066 => @ShannonSharpe
# 102071743 => @GovWhitmer