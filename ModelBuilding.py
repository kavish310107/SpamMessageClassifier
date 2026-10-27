import pandas as pd
import ast
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pickle

df = pd.read_csv(r'7. SpamDetectionClassifier\transformed_spam.csv')


# #found top k most commonly used words in spam messages
# lists = []
# for strg in df[df['target']==1]['text'].tolist() :
#     a = ast.literal_eval(strg)
#     for i in a :
#         lists.append(i)

# dicts_spam = Counter(lists).most_common(20)
# df2 = pd.DataFrame(dicts_spam)
# # print(dicts)
# # print(len(lists))

# sns.barplot(x=df2[0],y=df2[1])
# plt.xticks(rotation='vertical')
# plt.show()


#4. MODEL CREATION
# We start with creating model for Naive Bayes , as NB works best on text related data.
# To train the NB model , we need numerical quantities, which are not currently present , rn we have full text input , so we vectorize it to convert it to numerical columns

cv = CountVectorizer()
X = cv.fit_transform(df['text']).toarray()

tf = TfidfVectorizer(max_features=3000)
x = tf.fit_transform(df['text']).toarray()

y = df.iloc[:,0]

# print(X.shape)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=22)

# m1 = GaussianNB()
# m1.fit(xtrain,ytrain)
# ypred1 = m1.predict(xtest)
# print("Model m1 : ",classification_report(ypred1,ytest))

# m3 = BernoulliNB()
# m3.fit(xtrain,ytrain)
# ypred3 = m3.predict(xtest)
# print("Model m3 : ",classification_report(ypred3,ytest))

m4 = MultinomialNB()
m4.fit(xtrain,ytrain)
ypred4 = m4.predict(xtest)
print("Model m4 : ",classification_report(ypred4,ytest))
print(confusion_matrix(ypred4,ytest))


"""For spam classifier precision is more important, so we consider precision over recall. bernonulliNB(m3) gives best precision and accuracy score. Select that and try another vectorization method on it. Look up for tf-idf"""

"""for tf-idf with multinomial gives 100% precision ,so choosing that"""

"""Nitish sir tried same procedure on all other algorithms, tried to improve accuracy and precision by changing tf-idf max_features (3000) , then tried to create voting classifier based on top performing models from all trained models, then tried stacking,etc .Did all this for  different algo like DT,SVC,RF,Bagging,boosters,etc. Best outcome came for tf-idf with multinomialNB for tf-idf max_features 3000 only, so that is the final model.

I have not done all that because too much work and not capable to do it in vs code require collab/jupyter notebook for that.    """

"""SO FINALLY , OUT FINAL MODEL IS THE BINOMIAL NB  MODEL WITH TF-IDF VECTORIZATION. NOW WE CAN SAVE THE MODEL AND THE VECTORS"""

pickle.dump(tf,open('vectorizer.pkl','wb'))
pickle.dump(m4,open('model.pkl','wb'))

#saved the fitted ML model and vectorizer

# print(df.head())