from django.shortcuts import render

# Create your views here.
import pandas as pd #to handle our dataset
from sklearn.feature_extraction.text import CountVectorizer #for converting text data into numerical data
from sklearn.model_selection import train_test_split #to split data into training and testing
from sklearn.naive_bayes import MultinomialNB #the algorithm we use for spam detection
from sklearn.metrics import accuracy_score #for calculating data accuracy
from.forms import MessageForm

dataset =pd.read_csv('E:/IIT ROPAR/IIT GROUP/spam_detection_ML/spam detection/spamDetection/detector/emails.csv')
dataset.head()
vectorizer = CountVectorizer()
x= vectorizer.fit_transform(dataset['text'])
x_train,x_test, y_train, y_test = train_test_split(x, dataset['spam'],test_size=0.2) #80%training data 20% testine=g data

#train a naive_bayes classifier using our training data
model = MultinomialNB()
model.fit(x_train, y_train)


#function to predict if a massage spam or not
def predictMessage(message):
  messageVector = vectorizer.transform([message])
  prediction = model.predict(messageVector)
  return 'Spam' if prediction[0] == 1 else 'Ham'
#this function we make to render web page
def Home(request):
  result = None
  if request.method == 'POST':
    form = MessageForm(request.POST)
    if form.is_valid():
      message = form.cleaned_data['text']
      result = predictMessage(message)
  else:
    form = MessageForm()

  return render(request, 'home.html', {'form': form, 'result': result})

      
 
#get user input to predict

userMassage = input('Enter text to predict: ')
prediction = predictMessage(userMassage)
print("The message is: ",prediction)
