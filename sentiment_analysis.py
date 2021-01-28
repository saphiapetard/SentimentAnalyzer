import io
import random
import math
import numpy as np
import string
import nltk
from nltk.corpus import stopwords  # for stopwords
from nltk.metrics import confusionmatrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Step 1: loading data for agent
# opening files into np arrays
with io.open('rt-polarity.pos', 'r', encoding="ISO-8859-1") as fpos:
    pos = np.genfromtxt(fpos, delimiter='\n', dtype='str')
with io.open('rt-polarity.neg', 'r', encoding="ISO-8859-1") as fneg:
    neg = np.genfromtxt(fneg, delimiter='\n', dtype = 'str')

# Step 2: text-processing function
def process_text(pos, neg):
    # a) tokenizing sentence strings into lists of words
    tokenized_pos, tokenized_neg = [], []
    tokenizer1 = nltk.RegexpTokenizer(r'\w+')
    pos_list = list(pos)
    neg_list = list(neg)

    for i in range(pos.size):
        tokenized_sent1 = tokenizer1.tokenize(pos[i])
        tokenized_pos.append(tokenized_sent1)
        tokenized_sent2 = tokenizer1.tokenize(neg[i])
        tokenized_neg.append(tokenized_sent2)
    # now i have two lists, which store sentences as lists of words

    # b) lemmatizing, stopwords, frequency distribution
    lemma = nltk.stem.WordNetLemmatizer()

    stop_words = list(set(stopwords.words('english')))
    my_stopwords = ['the','a']
    stop_words.extend(my_stopwords)

    #creating full list for freq distribution
    flat_listP = [word for sentence in tokenized_pos for word in sentence]
    flat_listN = [word for sentence in tokenized_neg for word in sentence]
    full_list = flat_listP+flat_listN

    fdist = nltk.FreqDist(full_list)
    freq_list = []
    freq_tuples = fdist.most_common(500)
    for i in range(len(freq_tuples)):
        freq_list.append(freq_tuples[i][0])

    for sentence in tokenized_pos:
        for w in sentence:
            if w in freq_list:
                sentence.remove(w)
            elif w in stop_words or w in string.punctuation:
                sentence.remove(w)
            lemma.lemmatize(w)

    for sentence in tokenized_neg:
        for w in sentence:
            if w in freq_list:
                sentence.remove(w)
            elif w in stop_words or w in string.punctuation:
                sentence.remove(w)
            lemma.lemmatize(w)

    return tokenized_pos, tokenized_neg


# now i have a final, stemmed and tokenized lists for both pos and neg reviews [still separate]
#eg function call --> goodlist, badlist = process_text(pos,neg)#

# Step 3: Training and Test Data (80%-20% split)
# input = list of tokenized sentences x2
def train_and_test(pos_list, neg_list,which):
    # a) randomize order
    random.shuffle(pos_list)
    random.shuffle(neg_list)
    # b) set first 80% of sample to train, last 20% to test
    eighty_ind_p = math.floor(0.8*len(pos_list))
    eighty_ind_n = math.floor(0.8*len(neg_list))
    train_file_pos = pos_list[:eighty_ind_p]
    train_file_neg = neg_list[:eighty_ind_n]
    test_file_pos = pos_list[eighty_ind_p:]
    test_file_neg = neg_list[eighty_ind_n:]
    # c) depending on use, return accordingly
    if (which == 'train'):
        return train_file_pos,train_file_neg
    else:
        return test_file_pos, test_file_neg
# now i have access to pos/neg training lists, and pos/neg testing lists


# Step 4: Classifier-readable Data
    # a) function for making tokenized sentences into dictionaries for classifier readability
def word_feats(sentence):
    return dict([word,True] for word in sentence)

    # b) function for creating classifier-readable list of inputs and their labels
def make_data(pos, neg):
    pos_feats_l, neg_feats_l = [],[]
    for i in range(len(pos)):
        pos_feats = [word_feats(pos[i]), 'pos']
        pos_feats_l.append(pos_feats)
        neg_feats = [word_feats(neg[i]), 'neg']
        neg_feats_l.append(neg_feats)
    return pos_feats_l, neg_feats_l
# list of dictionary + label items

    # c) function for tracking correct labels, to use for model evaluation/comparison
def test_labels(test_p,test_n):
    p_labels = []
    n_labels = []
    for x in test_p:
        p_labels.append('pos')
    for x in test_n:
        n_labels.append('neg')

    return p_labels+n_labels

    # d) text-processing and data creation:
        # training set
sentences_p, sentences_n = process_text(pos,neg)
train_p, train_n = train_and_test(sentences_p,sentences_n,'train')
poseg, negeg = make_data(train_p,train_n)

trainfeats = poseg + negeg

        # testing set
test_p, test_n = train_and_test(sentences_p,sentences_n,'test')
postest, negtest = make_data(test_p,test_n)
testfeats = postest+negtest

    # correct labels of test set
labelled_test = test_labels(test_p,test_n)

# Step 5: Naive Bayes Classifier
classifierNB = nltk.NaiveBayesClassifier.train(trainfeats)

        # evaluation of classifier - counts wrong assignments
NBnomatch_count = 0
NBlabels = []
for i in range(len(testfeats)):
    classification = classifierNB.classify(word_feats(testfeats[i][0]))
    NBlabels.append(classification)
    if (classification != labelled_test[i]):
        NBnomatch_count = NBnomatch_count + 1

probability_corrNB = 1 - (NBnomatch_count / len(testfeats))
print("Using a Naive Bayes Classification model, the correct answer is found with probability " + str(probability_corrNB))
NBcm = confusionmatrix.ConfusionMatrix(NBlabels,labelled_test)
print("The Naive Bayes model can be evaluated by observing the following confusion matrix: \n")
print(NBcm)
print("\n\n")

# Step 6: Logistic Regression Classifier
classifierLR = nltk.classify.SklearnClassifier(LogisticRegression())
classifierLR.train(trainfeats)

LRno_match_count = 0
LRlabels = []

for i in range(len(testfeats)):
    classification = classifierLR.classify(word_feats(testfeats[i][0]))
    LRlabels.append(classification)
    if (classification != labelled_test[i]):
        LRno_match_count = LRno_match_count + 1
    # catch 'future warnings'

probability_corrLR = 1 - (LRno_match_count / len(testfeats))
print("Using a Logistic Regression model, the correct answer is found with probability " + str(probability_corrLR))
LRcm = confusionmatrix.ConfusionMatrix(LRlabels,labelled_test)
print("The Logistic Regression model can be evaluated by observing the following confusion matrix: \n")
print(LRcm)
print("\n\n")

# Step 7: Support Vector Machine Classifier
classifierSVM = nltk.classify.SklearnClassifier(LinearSVC())
classifierSVM.train(trainfeats)

SVMno_match_count = 0
SVMlabels = []
for i in range(len(testfeats)):
    classification = classifierSVM.classify(word_feats(testfeats[i][0]))
    SVMlabels.append(classification)
    if (classification != labelled_test[i]):
        SVMno_match_count = SVMno_match_count + 1

probability_corrSVM = 1 - (SVMno_match_count / len(testfeats))
print("Using a Support Vector Machine model, the correct answer is found with probability " + str(probability_corrSVM))
SVMcm = confusionmatrix.ConfusionMatrix(SVMlabels,labelled_test)
print("The SVM model can be evaluated by observing the following confusion matrix: \n")
print(SVMcm)
print("\n\n")

#Step 8: Random Label Generator (for comparison)
# randomly assigning either 0,1 (0 = pos, 1 = neg)
rand_list = []
for i in range(0,len(testfeats)):
    x = random.randint(0,1)
    if x == 0:
        rand_list.append("pos")
    else:
        rand_list.append("neg")

# checking for amount of correct results
corr_count = 0
for i in range (0, len(testfeats)):
    if rand_list[i] == labelled_test[i]:
        corr_count = corr_count + 1

probability_corrRL = corr_count / len(rand_list)
print("Using randomly generated labels, the correct answer is found with probability " + str(probability_corrRL))
Rcm = confusionmatrix.ConfusionMatrix(rand_list,labelled_test)
print("The random model can be evaluated by observing the following confusion matrix: \n")
print(Rcm)
print("\n\n")