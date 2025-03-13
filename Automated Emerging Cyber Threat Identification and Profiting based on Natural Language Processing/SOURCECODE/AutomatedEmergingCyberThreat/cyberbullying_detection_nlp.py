import os
from tkinter import *
import tkinter
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.feature_extraction.text import CountVectorizer

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,confusion_matrix



main = tkinter.Tk()
main.title("Automated Emerging Cyber Threat Identification and Profiting based on Natural Language Processing")
main.geometry("1300x1200")

global filename
global balance_data
global data
global X, Y, X_train, X_test, y_train, y_test
global LR_acc, NB_acc, RFT_acc,Xgb_acc,DT_acc


def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.insert(END, "Dataset loaded\n\n")


def preprocess():
    global filenam
    text.delete('1.0', END)
    balance_data = pd.read_csv(filename, nrows = 20000)
    text.insert(END, "Information about the dataset\n\n")
    text.insert(END, balance_data.head())
    text.insert(END, "\n\n")
    # Shape of dataframe
    text.insert(END, "Shape of dataframe\n\n")
    text.insert(END, balance_data.shape)
    text.insert(END, "\n\n")
    # Listing the features of the dataset
    text.insert(END, "Listing the features of the dataset\n\n")
    text.insert(END, balance_data.columns)
    text.insert(END, "\n\n")
    # Information about the dataset
    balance_data.info()
    # 1. Handling Null Values
    balance_data.isna().any()
    balance_data.isna().sum()
    """### 2. Handling Duplicate Values"""

    balance_data.nunique()

    balance_data['tweet_text'].nunique()

    """### 3. Class Distributions"""
    balance_data['cyberbullying_type'].value_counts()

    # Create a bar plot of the class distribution
    class_counts = balance_data['cyberbullying_type'].value_counts()
    class_counts.plot(kind='bar')
    plt.title('Class Distribution of Cyberbullying Types')
    plt.xlabel('Labels')
    plt.ylabel('Number of Tweets')
    plt.show()
    """### 4. Word Count"""

    from collections import Counter
    import re

    import nltk
    from nltk.corpus import stopwords

    # Concatenate all tweet texts into a single string
    all_text = ' '.join(balance_data['tweet_text'].values)
    # Remove URLs, mentions, and hashtags from the text
    all_text = re.sub(r'http\S+', '', all_text)
    all_text = re.sub(r'@\S+', '', all_text)
    all_text = re.sub(r'#\S+', '', all_text)
    # Split the text into individual words
    words = all_text.split()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]
    text.insert(END, "\n")
    text.insert(END, "Count the frequency of each word")
    # Count the frequency of each word
    word_counts = Counter(words)
    top_words = word_counts.most_common(100)
    text.insert(END,"\n")
    text.insert(END, top_words)
    text.insert(END, "\n")
    # Create a bar chart of the most common words
    top_words = word_counts.most_common(10)  # Change the number to show more/less words
    x_values = [word[0] for word in top_words]
    y_values = [word[1] for word in top_words]
    plt.bar(x_values, y_values)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Most Commonly Used Words')
    plt.show()
    text.insert(END, "Removed non numeric characters from dataset\n\n")
def SentimentAnalysis():
    text.delete('1.0', END)
    import pandas as pd
    import matplotlib.pyplot as plt
    from textblob import TextBlob
    global balance_data
    balance_data = pd.read_csv(filename, nrows =20000)

    # perform sentiment analysis on each text in DataFrame
    sentiment_scores = []
    for text1 in balance_data['tweet_text']:
        analysis = TextBlob(text1)
        sentiment_scores.append((analysis.sentiment.polarity, analysis.sentiment.subjectivity))
    text.insert(END, "\n")
    text.insert(END, "sentiment scores for all tweets")
    # create DataFrame with sentiment scores
    sentiment_df = pd.DataFrame(sentiment_scores, columns=['polarity', 'subjectivity'])
    text.insert(END, "\n")
    text.insert(END, sentiment_df)
    # plot distribution of sentiment scores
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    sentiment_df['polarity'].plot(kind='hist', ax=axes[0], title='Polarity')
    sentiment_df['subjectivity'].plot(kind='hist', ax=axes[1], title='Subjectivity')
    plt.show()
def nerModel():
    text.delete('1.0', END)
    global balance_data
    import spacy
    from spacy import displacy

    # sample text
    text1 = balance_data['tweet_text'].iloc[4]

    # load pre-trained NER model
    nlp = spacy.load('en_core_web_sm')

    # perform named entity recognition
    doc = nlp(text1)

    # visualize named entities
    displacy.render(doc, style='ent', jupyter=True)
    text.insert(END, "perform named entity recognition")
    text.insert(END, "\n")
    text.insert(END, doc)
    text.insert(END, "\n")

def posModel():
    import spacy
    from spacy import displacy
    text.delete('1.0', END)
    global balance_data
    # sample text
    text1 = balance_data['tweet_text'].iloc[1]

    # load pre-trained POS tagging model
    nlp = spacy.load('en_core_web_sm')

    # perform POS tagging
    doc = nlp(text1)

    # visualize POS tagging
    displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})
    text.insert(END, "\n")
    text.insert(END, "perform POS tagging")
    text.insert(END, "\n")
    text.insert(END, doc)
def topicModel():
    text.delete('1.0', END)
    global balance_data
    global X, Y, X_train, X_test, y_train, y_test
    import gensim
    # Preprocessing
    tokens = [[word for word in sentence.split()] for sentence in balance_data['tweet_text']]
    dictionary = gensim.corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    # Topic Modeling
    num_topics = 10
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    text.insert(END, "\n")
    text.insert(END, "Data Cleaning Process is started...")

    def clean_text(texta):
        # Remove HTML tags
        texta = re.sub('<.*?>', '', texta)

        # Remove non-alphabetic characters and convert to lowercase
        texta = re.sub('[^a-zA-Z]', ' ', texta).lower()

        # Remove URLs, mentions, and hashtags from the text
        texta = re.sub(r'http\S+', '', texta)
        texta = re.sub(r'@\S+', '', texta)
        texta = re.sub(r'#\S+', '', texta)

        # Tokenize the text
        words = nltk.word_tokenize(texta)

        # Remove stopwords
        words = [w for w in words if w not in stopwords.words('english')]

        # Stem the words
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]

        # Join the words back into a string
        texta = ' '.join(words)
        text.insert(END, "\n")
        text.insert(END, texta)
        return texta

    tqdm.pandas()
    balance_data['cleaned_text'] = balance_data['tweet_text'].progress_apply(clean_text)
    # Create the Bag of Words model
    cv = CountVectorizer()
    X = cv.fit_transform(balance_data['cleaned_text']).toarray()
    Y = balance_data['cyberbullying_type']

def generateModel():
    text.delete('1.0', END)
    global balance_data
    global X,Y,X_train, X_test, y_train, y_test
    """## 4. Splitting the Data:
    The data is split into train & test sets, 80-20 split.
    """
    # Splitting the dataset into train and test sets: 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    text.insert(END, "Train & Test Model Generated\n\n")
    text.insert(END, "Total Dataset Size : " + str(len(balance_data)) + "\n")
    text.insert(END, "Split Training Size : " + str(len(X_train)) + "\n")
    text.insert(END, "Split Test Size : " + str(len(X_test)) + "\n")



def plot_confusion_matrix(test_Y, predict_y):
 C = confusion_matrix(test_Y, predict_y)
 A =(((C.T)/(C.sum(axis=1))).T)
 B =(C/C.sum(axis=0))
 plt.figure(figsize=(20,4))
 labels = [1,2]
 cmap=sns.light_palette("blue")
 plt.subplot(1, 3, 1)
 sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Confusion matrix")
 plt.subplot(1, 3, 2)
 sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Precision matrix")
 plt.subplot(1, 3, 3)
 sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Recall matrix")
 plt.show()

def runRFT():
    text.delete('1.0', END)
    global RFT_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];
    text.insert(END, "Total Features : " + str(total) + "\n")
    from sklearn.ensemble import RandomForestClassifier

    # instantiate the model
    forest = RandomForestClassifier(n_estimators=10)

    # fit the model
    forest.fit(X_train, y_train)

    # predicting the target value from the model for the samples
    y_train_forest = forest.predict(X_train)
    y_test_forest = forest.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_forest = metrics.accuracy_score(y_train, y_train_forest)
    acc_test_forest = metrics.accuracy_score(y_test, y_test_forest)
    RFT_acc = acc_test_forest
    # computing the accuracy, f1_score, Recall, precision of the model performance
    text.insert(END, "\n")
    text.insert(END, "Random Forest : Accuracy on training Data: {:.3f}".format(acc_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Random Forest : Accuracy on test Data: {:.3f}".format(acc_test_forest))
    text.insert(END, "\n")

    f1_score_train_forest = metrics.f1_score(y_train, y_train_forest)
    f1_score_test_forest = metrics.f1_score(y_test, y_test_forest)
    text.insert(END, "\n")
    text.insert(END, "Random Forest : f1_score on training Data: {:.3f}".format(f1_score_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Random Forest : f1_score on test Data: {:.3f}".format(f1_score_test_forest))
    text.insert(END, "\n")

    recall_score_train_forest = metrics.recall_score(y_train, y_train_forest)
    recall_score_test_forest = metrics.recall_score(y_test, y_test_forest)
    text.insert(END, "\n")
    text.insert(END, "Random Forest : Recall on training Data: {:.3f}".format(recall_score_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Random Forest : Recall on test Data: {:.3f}".format(recall_score_test_forest))
    text.insert(END, "\n")

    precision_score_train_gbc = metrics.precision_score(y_train, y_train_forest)
    precision_score_test_gbc = metrics.precision_score(y_test, y_test_forest)
    text.insert(END, "\n")
    text.insert(END, "Random Forest : precision on training Data: {:.3f}".format(precision_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END, "Random Forest : precision on test Data: {:.3f}".format(precision_score_test_gbc))
    text.insert(END, "\n")
    # computing the classification report of the model
    text.insert(END, "\n")
    text.insert(END, "Classification Report: \n")
    text.insert(END, metrics.classification_report(y_test, y_test_forest))
    plot_confusion_matrix(y_train, y_train_forest)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_forest)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgreen', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()


def runLR():
    text.delete('1.0', END)
    global LR_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];

    text.insert(END, "Total Features : " + str(total) + "\n")
    from sklearn.linear_model import LogisticRegression
    # from sklearn.pipeline import Pipeline

    # instantiate the model
    log = LogisticRegression(max_iter = 1000)

    # fit the model
    log.fit(X_train, y_train)

    # predicting the target value from the model for the samples

    y_train_log = log.predict(X_train)
    y_test_log = log.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_svc = metrics.accuracy_score(y_train, y_train_log)
    acc_test_svc = metrics.accuracy_score(y_test, y_test_log)
    LR_acc=acc_test_svc
    # computing the accuracy, f1_score, Recall, precision of the model performance
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : Accuracy on training Data: {:.3f}".format(acc_train_svc))
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : Accuracy on test Data: {:.3f}".format(acc_test_svc))
    text.insert(END, "\n")

    f1_score_train_svc = metrics.f1_score(y_train, y_train_log)
    f1_score_test_svc = metrics.f1_score(y_test, y_test_log)
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : f1_score on training Data: {:.3f}".format(f1_score_train_svc))
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : f1_score on test Data: {:.3f}".format(f1_score_test_svc))
    text.insert(END, "\n")

    recall_score_train_gbc = metrics.recall_score(y_train, y_train_log)
    recall_score_test_gbc = metrics.recall_score(y_test, y_test_log)
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : Recall on training Data: {:.3f}".format(recall_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : Recall on test Data: {:.3f}".format(recall_score_test_gbc))
    text.insert(END, "\n")

    precision_score_train_gbc = metrics.precision_score(y_train, y_train_log)
    precision_score_test_gbc = metrics.precision_score(y_test, y_test_log)
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : precision on training Data: {:.3f}".format(precision_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END, "Logistic Regression : precision on test Data: {:.3f}".format(precision_score_test_gbc))
    text.insert(END, "\n")
    # computing the classification report of the model
    text.insert(END, "\n")
    text.insert(END, "Classification Report: \n")
    text.insert(END, metrics.classification_report(y_test, y_test_log))
    plot_confusion_matrix(y_train, y_train_log)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_log)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgreen', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()





def runDT():
    text.delete('1.0', END)
    global DT_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];

    text.insert(END, "Total Features : " + str(total) + "\n")

    from sklearn.tree import DecisionTreeClassifier

    # instantiate the model
    tree = DecisionTreeClassifier(max_depth=30)

    # fit the model
    tree.fit(X_train, y_train)

    # predicting the target value from the model for the samples

    y_train_tree = tree.predict(X_train)
    y_test_tree = tree.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_forest = metrics.accuracy_score(y_train, y_train_tree)
    acc_test_forest = metrics.accuracy_score(y_test, y_test_tree)
    DT_acc=acc_test_forest
    # computing the accuracy, f1_score, Recall, precision of the model performance
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : Accuracy on training Data: {:.3f}".format(acc_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : Accuracy on test Data: {:.3f}".format(acc_test_forest))
    text.insert(END, "\n")

    f1_score_train_forest = metrics.f1_score(y_train, y_train_tree)
    f1_score_test_forest = metrics.f1_score(y_test, y_test_tree)
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : f1_score on training Data: {:.3f}".format(f1_score_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : f1_score on test Data: {:.3f}".format(f1_score_test_forest))
    text.insert(END, "\n")

    recall_score_train_forest = metrics.recall_score(y_train, y_train_tree)
    recall_score_test_forest = metrics.recall_score(y_test, y_test_tree)
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : Recall on training Data: {:.3f}".format(recall_score_train_forest))
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : Recall on test Data: {:.3f}".format(recall_score_test_forest))
    text.insert(END, "\n")

    precision_score_train_gbc = metrics.precision_score(y_train, y_train_tree)
    precision_score_test_gbc = metrics.precision_score(y_test, y_test_tree)
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : precision on training Data: {:.3f}".format(precision_score_train_gbc))
    text.insert(END, "\n")
    text.insert(END, "Decision Tree : precision on test Data: {:.3f}".format(precision_score_test_gbc))
    text.insert(END, "\n")
    # computing the classification report of the model
    text.insert(END, "\n")
    text.insert(END, "Classification Report: \n")
    text.insert(END, metrics.classification_report(y_test, y_test_tree))
    plot_confusion_matrix(y_train, y_train_tree)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_tree)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgreen', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()





def graph():
    height = [LR_acc,RFT_acc,DT_acc]
    bars = ('LR Accuracy','RFT Accuracy','DT Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Explainable AI Techniques")
    plt.ylabel("Accuracy Score")
    plt.title("Comparison of Performance Estimation")
    plt.show()


font = ('times', 14, 'bold')
title = Label(main,
              text='AUTOMATED EMERGING CYBER THREAT INDENTICATION AND PROFITING BASED ON NATURAL LANGUAGE PROCESSING')
title.config(bg='pink', fg='brown')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=0)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Cyberbullying Tweets Dataset", command=upload)
upload.place(x=700, y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=700, y=150)

preprocess = Button(main, text="Dataset Pre-processing", command=preprocess)
preprocess.place(x=700, y=200)
preprocess.config(font=font1)


SentimentAna = Button(main, text="Sentiment Analysis", command=SentimentAnalysis)
SentimentAna.place(x=700, y=250)
SentimentAna.config(font=font1)

NER= Button(main, text="Named Entity Recognition (NER)", command=nerModel)
NER.place(x=700, y=300)
NER.config(font=font1)

pos= Button(main, text="Part-of-Speech (POS) Tagging", command=posModel)
pos.place(x=700, y=350)
pos.config(font=font1)

TopicModeling= Button(main, text="Topic Modeling Visualization", command=topicModel)
TopicModeling.place(x=700, y=400)
TopicModeling.config(font=font1)

model = Button(main, text="Feature Extraction", command=generateModel)
model.place(x=700, y=450)
model.config(font=font1)

runsvm = Button(main, text="Logistics Regression Algorithm", command=runLR)
runsvm.place(x=700, y=500)
runsvm.config(font=font1)

rundet = Button(main, text="Decision Tree Classifier Algorithm", command=runDT)
rundet.place(x=700, y=550)
rundet.config(font=font1)

runsvm = Button(main, text="Random Forest Algorithm", command=runRFT)
runsvm.place(x=700, y=600)
runsvm.config(font=font1)


graphButton = Button(main, text="Comparision of Models", command=graph)
graphButton.place(x=700, y=650)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=80)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=100)
text.config(font=font1)

main.config(bg='brown')
main.mainloop()
