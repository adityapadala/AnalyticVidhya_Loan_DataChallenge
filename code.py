import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
%matplotlib inline
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

#importing training and testing datasets
train = pd.DataFrame
test = pd.DataFrame
train=pd.read_csv('fakepath/train_u6lujuX.csv')
test=pd.read_csv('fakepath/test_Y3wMUE5.csv')

#checking for missing values
train.isnull().sum()
test.isnull().sum()

def Missing_values(data):
    #missing value: to check for null values in each column
    data.isnull().sum()

    #gender,married,Self employed
    data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
    data['Married'].fillna(data['Married'].mode()[0],inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace=True)

    #to find the missing data using apply fun
    def is_missing(a):
        return sum(a.isnull())

    #missing in columns
    print("Printing the missing columns")
    print(data.apply(is_missing))

    #imputing loanamount
    impute_grps = data.pivot_table(values = ['LoanAmount'], index = ['Gender','Married','Self_Employed'] , aggfunc = np.mean)
    print("LoanAmount grouped by gender married self_employed")
    print(impute_grps)

    #print(impute_grps.loc['Female','No','No'][0])
    for i,row in data.loc[data['LoanAmount'].isnull()].iterrows():
        a = tuple([row['Gender'],row['Married'],row['Self_Employed']])
        data['LoanAmount'].fillna(impute_grps.loc[a][0],inplace=True)
        
    #imputing Dependents missing values
    impute_dep = data.groupby(['Gender','Married'])['Dependents'].agg(['count'])
    
    for i,row in data.loc[data['Dependents'].isnull()].iterrows():
        data['Dependents'].fillna(impute_dep.loc[row['Gender']].loc[row['Married']][0],inplace=True)

    #imputing Loan Amount Term
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace=True)

    #imputing Credit_History
    data.pivot_table(values=['ApplicantIncome'],index=['Credit_History'], aggfunc = np.mean)
    data['Credit_History'].value_counts()

    knn_train=pd.DataFrame
    knn_test=pd.DataFrame
    knn_train = data[-data['Credit_History'].isnull()][['ApplicantIncome','CoapplicantIncome','LoanAmount','Credit_History']]
    knn_test = data[data['Credit_History'].isnull()][['ApplicantIncome','CoapplicantIncome','LoanAmount']]

    training, validation = train_test_split(knn_train, test_size = 0.2)

    #another way to split the dataframe
    #df = pd.DataFrame(np.random.randn(knn_train.count()[0], 1))
    #print(df)
    #msk = np.random.rand(len(df)) < 0.8
    #print(msk)
    #train = knn_train[msk]
    #test = knn_train[~msk]
    #print(train.count()[0])
    #print(test.count()[0])

    def normal(x):
        return x/(max(x)-min(x))
    #normalizing the numerical values for knn classification    
    training=training.apply(normal)
    validation=validation.apply(normal)

    for i in range(1,21):
        knn  = KNeighborsClassifier(n_neighbors=i)
        knn.fit(training[['ApplicantIncome','CoapplicantIncome','LoanAmount']], training['Credit_History'])
        predictions = knn.predict(validation[['ApplicantIncome','CoapplicantIncome','LoanAmount']])
        print(i, (pd.crosstab(predictions,validation['Credit_History'],margins = True)[0][0]+pd.crosstab(predictions,validation['Credit_History'],margins = True)[1][1])/113)
        
    knn_train = knn_train.apply(normal)
    knn_test = knn_test.apply(normal)
    knn  = KNeighborsClassifier(n_neighbors=1)
    knn.fit(knn_train[['ApplicantIncome','CoapplicantIncome','LoanAmount']], knn_train['Credit_History'])
    predictions = knn.predict(knn_test)


    #predictions[0:50].tolist()
    credit_pred=pd.DataFrame(predictions.tolist(),index=knn_test.index)

    i=0
    pred = list(predictions)
    for boolValue in data['Credit_History'].isnull():
        if boolValue == True:
            data['Credit_History'][i] = pred[0]
            pred.pop(0)
        else:
            pass
        i+=1

    pd.concat([data[data['Credit_History'].isnull()][['Credit_History']],credit_pred],axis=1)
    print(data.isnull().sum())
    return data

	
def outlier(data):
    data['ApplicantIncome'].describe()
    plt.plot(data['ApplicantIncome'])
    data['ApplicantIncome'][data['ApplicantIncome'] > 10000]=10000
    data['ApplicantIncome'][data['ApplicantIncome'] < 500]=500
    data['CoapplicantIncome'].describe()
    data['CoapplicantIncome'][data['CoapplicantIncome'] > 4500]=4500
    data['LoanAmount'].describe()
    plt.boxplot(data['LoanAmount'])
    data['LoanAmount'][data['LoanAmount'] > 250]=250
    return data
	
	
train1=pd.DataFrame()
test1=pd.DataFrame()
train1=Missing_values(train)
test1=Missing_values(test)
train1=outlier(train1)
test1=outlier(test1)

features = train1.copy()
df1 = features[['Gender','Married','Dependents','Education','Self_Employed','Loan_Amount_Term','Credit_History'
          ,'Property_Area','Loan_Status']].apply(lambda x: pd.factorize(x)[0])
df2 = features[['ApplicantIncome','CoapplicantIncome','LoanAmount']]
frames = [df1, df2]
train2=pd.concat(frames,axis=1)
f = train2.drop('Loan_Status', axis=1)
X = f.values.astype(np.float32)
y = (train1['Loan_Status'].values == 'Y').astype(np.int32)

features_test = test1.copy()
df3 = features_test[['Gender','Married','Dependents','Education','Self_Employed','Loan_Amount_Term','Credit_History'
          ,'Property_Area']].apply(lambda x: pd.factorize(x)[0])
df4 = features_test[['ApplicantIncome','CoapplicantIncome','LoanAmount']]
frames_test = [df3, df4]
test2=pd.concat(frames_test,axis=1)
test2_sub=test2.values.astype(np.float32)

#crearing a training and testing dataset for private evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
	
	
#decisiontreeclassification
clf = DecisionTreeClassifier(max_depth=8)
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
print("ROC AUC Decision Tree: {:.4f} +/-{:.4f}".format(np.mean(scores), np.std(scores)))
#ROC AUC Decision Tree: 0.6902 +/-0.0259


#RandomForestClassification
clf = RandomForestClassifier(n_estimators=100, max_features=4,max_depth=10)
scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc',n_jobs=-1)
print("ROC Random Forest: {:.4f} +/-{:.4f}".format(np.mean(scores), np.std(scores)))
#ROC Random Forest: 0.7660 +/-0.0663


#gradientBoostingClassification
param_grid = { 'learning_rate': [0.1,0.05,0.02,0.01],'max_depth':[4,6],'min_samples_leaf':[3,5,9,17],'max_features':[1.0,0.3,0.1]}
clf = GradientBoostingClassifier(n_estimators=100)
gs_cv = GridSearchCV(clf,param_grid).fit(X, y)
print (gs_cv.best_params_)

#adaboostingclassification
clf = AdaBoostClassifier(n_estimators=500, learning_rate=1.0)
scores = cross_val_score(clf, X, y, scoring='roc_auc', n_jobs=-1)
print("ROC Random Forest: {:.4f} +/-{:.4f}".format( np.mean(scores), np.std(scores)))

#fitting the best model
clf = clf.fit(X, y)
#misclassification rate on training set
sum(clf.predict(X_test)==y_test)/len(y_test)
x=clf.predict(test2_sub)

sub = pd.concat([test1['Loan_ID'],pd.DataFrame(x)],axis=1)
sub.columns=['Loan_ID','Loan_Status']
sub.to_csv('fakepath/submission.csv')
