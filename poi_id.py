#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("../tools/")

#For Plots 
import matplotlib.pyplot as plt


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler





#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','exercised_stock_options','bonus','shared_receipt_with_poi','fraction_emails_from_poi_to_person','fraction_emails_from_person_to_poi',
                 'total_stock_value','expenses','long_term_incentive'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#-------------------------------------------------------------------------
    
### Task 2: Remove outliers
    data_dict.pop('TOTAL')

#-------------------------------------------------------------------------
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
count_persons=0
count_poi = 0 
#pprint.pprint(data_dict)
for ii in my_dataset:
    count_persons += 1
    if my_dataset[ii]['poi']== 1.0:
        count_poi +=1
    if my_dataset[ii]['bonus'] != 'NaN' and my_dataset[ii]['bonus']> 0.4 *10**7 and my_dataset[ii]['salary']> 10**6:
        print 'Outlier_Person:' + ii

#    if ii.find('LOEHR') > 1:
#        print ii + 'Exception'
print 'Number of Persons in the data set: ' + str(count_persons)
print 'Number of POIs in the data set: ' + str(count_poi)
print 'Number of Features:' + str(len(my_dataset[ii]))

#New feature: fraction of emails_from_poi_to_this_person + fraction of emails_from_this_person_to_poi
for ii in my_dataset:
    if my_dataset[ii]['to_messages']!= 'NaN' and my_dataset[ii]['from_this_person_to_poi']!= 'NaN':
        my_dataset[ii]['fraction_emails_from_person_to_poi'] = float(my_dataset[ii]['from_this_person_to_poi'])/my_dataset[ii]['from_messages']
    else:
        my_dataset[ii]['fraction_emails_from_person_to_poi']='NaN'
    if my_dataset[ii]['from_messages']!= 'NaN' and my_dataset[ii]['from_poi_to_this_person']!= 'NaN':
        my_dataset[ii]['fraction_emails_from_poi_to_person'] = float(my_dataset[ii]['from_poi_to_this_person'])/my_dataset[ii]['to_messages']
    else:
        my_dataset[ii]['fraction_emails_from_poi_to_person']='NaN'
    #print my_dataset[ii]
    


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Scaling my features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

#pprint.pprint(labels)
#pprint.pprint(features)
#-------------------------------------------------------------------------
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree   import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics ,feature_selection

clf_Gaussian = GaussianNB()
tree = DecisionTreeClassifier(random_state=2)
tree2 = DecisionTreeClassifier(random_state=2)
clf_SVM = SVC(kernel = 'rbf', C = 1000.0)

#Selecting only the best features
#kbest = feature_selection.SelectKBest(k=5)

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def kbestfeatures(features,labels):
    #Selecting only the best features
    kbest = feature_selection.SelectKBest(k=7)
    kbest.fit(features,labels)
    #print 'Score using KBest'
    #print kbest.scores_
    return kbest.transform(features)
 


#-------------------------------------------------------------------------
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn import grid_search
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit
#from sklearn.cross_validation import StratifiedShuffleSplit
features_train, features_test, labels_train, labels_test = \
    train_test_split(kbestfeatures(features,labels), labels, test_size=0.3, random_state=42)
    
"""
### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]
"""
    


    
#Checking training set for outliers

clf_Gaussian.fit(features_train,labels_train)

parameters = {'min_samples_split':(2,10,100),'max_features':('sqrt','log2',None),}
#parameters = {'min_samples_split':(2,10,100)}
#Decision_Tree with GridSearch
clf_tree = grid_search.GridSearchCV(tree2,parameters)

clf_tree.fit(features_train,labels_train)
tree.fit(features_train,labels_train)

clf_SVM.fit(features_train,labels_train)

#Usinf SelectKBest


#Scatter Plot for the training data
usecolour = {'0' : 'blue', '1' : 'green'}
count =0
"""
for f1,f2,f3,f4,f5,f6,f8,f9,f10 in features_train:
    #if labels_train[count] == 1.0:
    if labels_train[count] == 1.0:
        green_dot = plt.scatter(f5,f6, c= 'green')
    else:
        blue_dot = plt.scatter(f5,f6, c= 'blue')
    count +=1
#plt.xlim((0),(1200000))
#plt.ylim((-100000),(.5*10**7))
plt.xlabel('fraction_emails_from_poi_to_person')
plt.ylabel('fraction_emails_from_person_to_poi')
plt.legend([blue_dot,green_dot],['Non-POI','POI'], loc= 'upper left')
#plt.savefig('feature_fraction_poi.png')
"""
print 'Score Naive_Bayes: \n' + str(clf_Gaussian.score(features_test,labels_test))
prediction1 = clf_Gaussian.predict(features_test)
#Recall and Prediction
recall = metrics.recall_score(labels_test,prediction1)
print 'recall:' + str(recall)
precision = metrics.precision_score(labels_test,prediction1)
print 'precision' + str(precision)
print 'f1 score: ' + str(metrics.f1_score(labels_test,prediction1))

# Decision Tree with GridSearch
print 'Decision Tree with GridSearch: \n' + str(clf_tree.score(features_test,labels_test))
prediction_tree1 = clf_tree.predict(features_test)

#Recall and Prediction
recall = metrics.recall_score(labels_test,prediction_tree1)
print 'recall:' + str(recall)
precision = metrics.precision_score(labels_test,prediction_tree1)
print 'precision' + str(precision)
print 'f1 score: ' + str(metrics.f1_score(labels_test,prediction_tree1))

# Decision Tree without GridSearch
print 'Decision Tree: \n' + str(tree.score(features_test,labels_test))
prediction_tree2 = tree.predict(features_test)

#Recall and Prediction
recall = metrics.recall_score(labels_test,prediction_tree2)
print 'recall:' + str(recall)
precision = metrics.precision_score(labels_test,prediction_tree2)
print 'precision' + str(precision)

print 'f1 score: ' + str(metrics.f1_score(labels_test,prediction_tree2))


print 'Feature importance:' 
pprint.pprint(tree.feature_importances_)


# SVM
print 'SVM: \n' + str(clf_SVM.score(features_test,labels_test))
prediction_SVM = clf_SVM.predict(features_test)

#Recall and Prediction
recall = metrics.recall_score(labels_test,prediction_SVM)
print 'recall:' + str(recall)
precision = metrics.precision_score(labels_test,prediction_SVM)
print 'precision' + str(precision)

print 'f1 score: ' + str(metrics.f1_score(labels_test,prediction_SVM))




#Using StratifiedShuffleSplit to tune Gaussian NB Classifier

cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
mean_accuracy = 0.0
mean_recall = 0.0
mean_precision = 0.0
mean_f1 = 0.0
features=kbestfeatures(features,labels)

for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        clf_SVM.fit(features_train,labels_train)
        prediction = clf_SVM.predict(features_test)
        mean_accuracy += clf_SVM.score(features_test,labels_test)
        mean_recall += metrics.recall_score(labels_test,prediction)
        mean_precision += metrics.precision_score(labels_test,prediction)
        mean_f1 += metrics.f1_score(labels_test,prediction)
print 'Score after applying StratifiedShuffleSplit:', mean_accuracy/1000
print 'Recall:', mean_recall/1000
print 'Precision:', mean_precision/1000
print 'F1 Score:', mean_f1/1000
print '\n'
        
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.



dump_classifier_and_data(clf_Gaussian, my_dataset, features_list)


import tester


tester.test_classifier(clf_Gaussian,my_dataset,features_list)
