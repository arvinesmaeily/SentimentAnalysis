# pandas is used for reading and writing csv files
import pandas
# CountVectorizer is used for converting a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
# MultinomialNB is a Naive Bayes classifier used for multinomial models
from sklearn.naive_bayes import MultinomialNB
# Imported methods are used for statistics about performance of ML
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("Step 3 of 3")

# import train and test dataframes
train_dataframe = pandas.read_csv("Train_DataFrame.csv", encoding='latin-1')
test_dataframe = pandas.read_csv("Test_DataFrame.csv", encoding='latin-1')

# separation of dataset items by labels
print("Separating train and test datasets...")
x_train = train_dataframe["Comment"].tolist()
x_test = test_dataframe["Comment"].tolist()
y_train = train_dataframe["State"].tolist()
y_test = test_dataframe["State"].tolist()

# vectoring comment string to a matrix of token counts
print("Vectoring comments...")
cvectorizer = CountVectorizer()
x_train_vector = cvectorizer.fit_transform(x_train)
x_test_vector = cvectorizer.transform(x_test)


# classifying multinomial models using Naive Bayes classifier
print("Classifying multinomial models using Naive Bayes classifier...")
MNB = MultinomialNB()
MNB.fit(x_train_vector, y_train)

# prediction for test dataset
print("Predicting results...")
predict = MNB.predict(x_test_vector)

# printing performance statistics
print("Calculating statistics...")
Accuracy = accuracy_score(predict, y_test)
Classification_Report = classification_report(predict, y_test)
Confusion_Matrix = confusion_matrix(predict, y_test)
print("Accuracy: ", Accuracy)
print("Classification Report: \n", Classification_Report)
print("Confusion Matrix: \n", Confusion_Matrix)
