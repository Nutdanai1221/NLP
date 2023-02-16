import twint

def scrp_twitter(word):
    c = twint.Config()
    c.Search = word
    c.Limit = 50
    c.Popular_tweets = True
    c.Store_object = True
    c.Hide_output = False # don't show output in terminal
    c.Lang = "en" # english
    twint.run.Search(c)
    tweets = twint.output.tweets_list

    return tweets
