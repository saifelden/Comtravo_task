import sys
import numpy as np
import json as js
import pandas as pd
import random
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint

class Email_Labels_Recongizer:
    def Read_Data(self,path):
		data=None
		with open(path) as data_file:    
			data = js.load(data_file)
		return data

    def get_doc_size(self,doc): 
	    #get the size of an email by check the start index of the last token and and the last token length
		doc_size=doc['tokens'][-1]['start']
		doc_size+=doc['tokens'][-1]['length']
		return doc_size

    def get_features(self,data):
		features=[]
		for doc in data:
			for word in doc['tokens']: 
				if word['rner'] not in features: #get all set of rner types eg:'B-Breakfast','I-Location'
					features.append(word['rner'])
				if word['where'] not in features:  #get all set of where types which is only subject','body'
					features.append(word['where'])
		features.append('length') #adding the length of the document or the email as features
		return features


    def calc_doc_features(self,doc,all_features):
        tokens=doc['tokens']
        doc_features={}
        for feature in all_features:
            doc_features[feature]=0
        for word in tokens:
            doc_features[word['rner']]+=1
            doc_features[word['where']]+=1
        doc_features['length']=self.get_doc_size(doc)
        return doc_features

    def Random_split(self,data):
    	random.shuffle(data)
    	Traing=data[0:6500]
    	Testing=data[6500:7200]
    	validation=data[7200:len(data)]
    	return Traing,validation,Testing

    def get_data_features(self,Traing,validation,Testing):
    	self.all_features=self.get_features(Traing)
    	Training_features=[]
    	validation_features=[]
    	Testing_features=[]
    	for doc in Traing:
    		Training_features.append(self.calc_doc_features(doc,self.all_features))
    	for doc in Testing:
    		Testing_features.append(self.calc_doc_features(doc,self.all_features))
    	for doc in validation:
    		validation_features.append(self.calc_doc_features(doc,self.all_features))

    	return Training_features,validation_features,Testing_features

    def get_labels(self,data):
    	labels=[]
    	for doc in data:
    		labels.append(doc['labels'])
    	return labels

    def Train_linear_SVC(self,features,labels): #train Multi Label Data with Linear Support Vector Classifier
    	SVC_model=OneVsOneClassifier(LinearSVC(random_state=0))
    	multi_target = MultiOutputClassifier(SVC_model, n_jobs=-1)
    	multi_target.fit(self.training_features, self.trainging_labels)
    	return multi_target


    def Train_Random_Forest(self,features,labels,number_of_trees):
    	Random_forest_model=RandomForestClassifier(n_estimators=number_of_trees)
    	rfmulti_target = MultiOutputClassifier(Random_forest_model, n_jobs=-1)
    	rfmulti_target.fit(self.training_features, self.trainging_labels)
    	return rfmulti_target


    def Calc_Accuracy(self,x,y):
    	ise=True
    	for i in range(len(x)):
            if x[i]!=y[0][i]:
                ise=False
                break
        return ise

    def to_Pandas_DataFrame(self,trainging_labels,training_features,Testing_labels,Testing_features,valid_labels,valid_features):
        self.trainging_labels=pd.DataFrame(trainging_labels).fillna(0)
        self.training_features=pd.DataFrame(training_features).fillna(0)

        self.testing_features=pd.DataFrame(Testing_features).fillna(0)
        self.testing_labels=pd.DataFrame(Testing_labels).fillna(0)
        self.validation_labels=pd.DataFrame(valid_labels).fillna(0)
        self.validation_features=pd.DataFrame(valid_features).fillna(0)


    def choose_best_number_of_trees(self,numlist):
    	best_num=None
    	best_accuracy=0
    	for num in numlist:
    		model=self.Train_Random_Forest(self.training_features,self.trainging_labels,num)
    		test_size=len(self.validation_labels.values)
    		curr_accuracy=0
    		for i in range(0,test_size):
    			x= self.validation_labels.values[i]
    			y= model.predict(self.validation_features.values[i].reshape(1,-1))
    			if self.Calc_Accuracy(x,y):
    				curr_accuracy+=1
    		if curr_accuracy > best_accuracy:
    			best_accuracy=curr_accuracy
    			best_num=num
    	return num


    def Test_Accuracy(self,model):
    	accuracy=0.0
    	test_size=len(self.testing_labels.values)+0.0
    	for i in range(0,int(test_size)):
    		x= self.testing_labels.values[i]
    		y= model.predict(self.testing_features.values[i].reshape(1,-1))
    		if self.Calc_Accuracy(x,y):
    			accuracy+=1.0
    	return accuracy/test_size

    def predict_missing_Labels(self,Unlabeled_features,model,Unlabeled):
        predict_unlabeled=model.predict(Unlabeled_features)
        self.labels=['booking','cancelation','issues','negotiation','other','rebooking']
        for i in range(len(predict_unlabeled)):
            Unlabeled[i]['labels']={}
            for j in range(len(predict_unlabeled[i])):
                if predict_unlabeled[i][j]==1:
                    Unlabeled[i]['labels'][self.labels[j]]=1
        return Unlabeled

    def predict_missing_Labels_with_Probability(self,Unlabeled_features,model,Unlabeled):
        predict_unlabeled=model.predict_proba(Unlabeled_features)
        self.labels=['booking','cancelation','issues','negotiation','other','rebooking']
        for i in range(len(predict_unlabeled)):
            Unlabeled[i]['labels']={}
            for j in range(len(predict_unlabeled[i][1])):
                if predict_unlabeled[i][1][j] >= 0.5:
                    Unlabeled[i]['labels'][self.labels[j]]= predict_unlabeled[i][1][j]
        return Unlabeled

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("predict_category <train_file> <test_file>")

    train_file, test_file = sys.argv[1:]

    Email_labeling=Email_Labels_Recongizer()
    print 'Reading Labeled Data'
    Labeled_Data=Email_labeling.Read_Data(train_file)
    print 'Reading Unlabeled Data'
    Unlabeled_Data=Email_labeling.Read_Data(test_file)
    print 'Split the labeled Data into Training set, Validation set and Test set and extract the features from the Data,please make sure that your Data is more than 7400 record'

    Training_Data,Validation_Data,Test_Data=Email_labeling.Random_split(Labeled_Data)

    Training_features,Validation_features,Testing_features = Email_labeling.get_data_features(Training_Data,Validation_Data,Test_Data)
    print 'Extracted features is :'
    print Email_labeling.get_features(Labeled_Data)

    print 'extract labels for the Training set, Validation set and Test set'

    Training_labels=Email_labeling.get_labels(Training_Data)
    Testing_labels=Email_labeling.get_labels(Test_Data)
    Validation_labels=Email_labeling.get_labels(Validation_Data)

    print 'Convert the Data to Pandas DataFrame'

    Email_labeling.to_Pandas_DataFrame(Training_labels,Training_features,Testing_labels,Testing_features,Validation_labels,Validation_features)

    
    print ''
    print 'To run the Random Forest Classifier it need number of Trees, I precalculated the best value for the number'
    print 'Trees to use or we can reevaluate the best number of trees by the vlaidation set.'

    print 'If you want to use the Precalculated Number of trees value press 1, press 2 if you want evaluate the number of trees'

    x=raw_input()
    numlist=[3,5,7,10]
    best_num=None
    if x == '1':
    	best_num=5
    elif x == '2':
    	best_num=Email_labeling.choose_best_number_of_trees(numlist)
    else:
    	print 'not a valid value'
    	best_num=5

    print 'Start Training ..'
    Model=Email_labeling.Train_Random_Forest(Training_features,Training_labels,best_num)

    #accuracy=Email_labeling.Test_Accuracy(Model)
    #print 'the Model Accuracy is: '+str(accuracy)

    print 'Start Filling the Missing Labels in the Unlabeled_Data'

    print 'If you want to fill missing labels without probaility press 1, press 2 if you want to view the probaility of a label'

    x=raw_input()
    if x=='1':
        unlabeled=[]
        for doc in Unlabeled_Data:
            unlabeled.append(Email_labeling.calc_doc_features(doc,Email_labeling.all_features))
        Unlabeled_DataFrame=pd.DataFrame(unlabeled).fillna(0)
    	Unlabeled_Data=Email_labeling.predict_missing_Labels(Unlabeled_DataFrame,Model,Unlabeled_Data)
        with open('comtravo_predictions.json', 'w') as outfile:
            js.dump(Unlabeled_Data, outfile)
    elif x=='2':
        unlabeled=[]
        for doc in Unlabeled_Data:
            unlabeled.append(Email_labeling.calc_doc_features(doc,Email_labeling.all_features))
        Unlabeled_DataFrame=pd.DataFrame(unlabeled).fillna(0)
        Unlabeled_Data=Email_labeling.predict_missing_Labels_with_Probability(Unlabeled_DataFrame,Model,Unlabeled_Data)
        with open('comtravo_predictions_probability.json', 'w') as outfile:
            js.dump(Unlabeled_Data, outfile)
    else:
    	print 'Not a valid input'



    


















