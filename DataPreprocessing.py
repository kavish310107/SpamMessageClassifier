import pandas as pd
import nltk 
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
from pathlib import Path

DATA_PATH = Path(__file__).with_name('transformed_spam.csv')

#2. EDA
#figuring out how word count affects spam/not spam messages
if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    df['char'] = df['text'].apply(len)
    df['words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
    df['sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

# sns.pairplot(df)
# corr_matrix = df[['target','char','sentences','words']].corr()
# sns.heatmap(corr_matrix,annot=True)
# sns.histplot(df[df['target']==0]['char'])
# sns.histplot(df[df['target']==1]['char'],color='red')
# plt.show()
#from pairplot , I see that char,words and sentences have almost linear relationship with few outliers
#from correlation matrix,  see that char,words and sentence are highly dependent on each other, i.e multicollinear. So in final prediction will use only 1 among them, also char has highest correlation with target, i.e 0.38 , so going to use that
#from histplot, I see that the number of characters for spam messages is much more than the number of characters for non spam messages, so can train model on that


#3. TEXT PREPROCESSING
ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))
string_punc = string.punctuation

def transform_text(text) :
    #reduce evrything to lower case
    text = text.lower()
    
    #convert string to list of words (tokenization)
    text = nltk.word_tokenize(text)
    #this does the same thing as function created by using text.split, gives list as output
    
    #remove special characters 
    y = []
    for i in text :
        if(i.isalnum()) :
            y.append(i)
    text = y[:]
    y.clear()
    
    #remove punctuations and stop words
    for i in text : 
        if i not in stopwords_set and i not in string_punc :
            y.append(i)
    
    #reduce all words to base form (stemming)
    lists = []
    for i in y :
        lists.append(ps.stem(i))
    
    return lists

if __name__ == '__main__':
    df['text'] = df['text'].apply(transform_text)

    df.to_csv(DATA_PATH, index=False)
