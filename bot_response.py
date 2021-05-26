import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
# import matplotlib.pyplot as plt

# reading text from a file and converting them to lower case with removal of punctuations


def read_line(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    return cleaned_text


# tokenizing the text


def preprocess_line(cleaned_text):
    tokenized_words = word_tokenize(cleaned_text, "english")

    # stop_words addition
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)

    return final_words

# identifying emotion


def emotion_detect(sentences):
    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')

            if word in sentences:
                emotion_list.append(emotion)
    return emotion_list


def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    # print(score)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        sentiment = "negative"
    elif pos > neg:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return sentiment



# type of text analysis
posts = nltk.corpus.nps_chat.xml_posts()[:10000]


def dialogue_act_features(post):
    features = {}
    for word in word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


def bot_response(sentiment):
    pos = ["Well said.", "You are thinking in the right direction.", "Very well thought.", "That’s great.",
           "It sounds like you really know what the problems are.", "Nice, let’s talk about that a little.",
           "It seems like you’ve been thinking about this a lot."]
    neu = ["That’s okay.", "Okay!", "Okay, let’s keep going.", "That’s really understandable. "]
    neg = ["I understand how you feel.", "I see how that can be frustrating.", "That must be really frustrating."
           , "This is really bothering you."]
    if sentiment == "negative":
        return random.choice(neg)
    elif sentiment == "positive":
        return random.choice(pos)
    else:
        return random.choice(neu)


"""
# Plotting the emotions on the graph

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()
"""

# processing sentence to get emotions/sentiments and bots response


def sentence_processing(text):
    cleaned_text = read_line(text)
    # count number of words in response
    words = cleaned_text.split()
    word_len = len(words)
    if word_len <= 25:
        final_words = preprocess_line(cleaned_text)
        # print(emotion_detect(final_words))
        w = Counter(emotion_detect(final_words))
        # print(w)
        # sentiment analysis
        sentiment = sentiment_analyse(cleaned_text)
        bot_res1 = bot_response(sentiment)
    else:
        long_res = ["It sounds like you really know what the problems are.", "Nice, let’s talk about that a little.",
                    "It seems like you’ve been thinking about this a lot.",
                    "Struggling with communication is difficult"]
        bot_res1 = random.choice(long_res)

    # classification of type of sentence
    featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # print(nltk.classify.accuracy(classifier, test_set))
    sentence_type = classifier.classify(dialogue_act_features(text))
    if sentence_type == 'whQuestion':
        bot_res2 = "Let's focus more on the session."

    return bot_res1

"""
if __name__=="__main__":
    text = input("User: ")
    print(sentence_processing(text))
    """



