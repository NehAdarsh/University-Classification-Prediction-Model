#!/usr/bin/env python
# coding: utf-8

# ### Step 1 Importing Required modules

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd


# ### Step 2. Import the Dataset

# In[111]:


college_data = pd.read_csv("/Users/abhinavadarsh/Desktop/NEHA/WInter_2ndQuarter/ALY6015/W3/College.csv")


# ### Step 3. Exploratory Data Analysis

# In[112]:


college_data.shape    #rows and column


# In[54]:


college_data.columns


# In[55]:


college_data.head()          


# In[56]:


college_data.describe()         #Basic descriptive statistics


# In[57]:


college_data["Private"].groupby(by = college_data["Private"]).count()    #Private column values


# In[58]:


#NA/Null values
missing_val = college_data.isna().sum()
null_val = college_data.isnull().sum()
print(missing_val, null_val)


# In[113]:


#assigning all variables except target to a new variable names Predictors
predictors = college_data[['Apps', 'Accept', 'Enroll', 'Top10perc',
       'Top25perc', 'F_Undergrad', 'P_Undergrad', 'Outstate', 'Room_Board',
       'Books', 'Personal', 'PhD', 'Terminal', 'S_F_Ratio', 'perc_alumni',
       'Expend', 'Grad_Rate']]

#assigning Private to a new variable named 'target
#y = college_data.Private
target = college_data.Private


# In[60]:


#Distribution of target variable
target.hist()
plt.suptitle('Target variable distribution')


# In[11]:


sns.pairplot(college_data[['Top10perc', 'Top25perc', 'Room_Board', 'Outstate', 'Books', 'Personal', 
                           'PhD', 'Terminal', 'perc_alumni', 'Grad_Rate', 'Expend', 'S_F_Ratio']])


# In[12]:


#Relationship between target and predictor variables
rel2 = sns.pairplot(college_data, x_vars=['Top10perc', 'Top25perc', 'Room_Board', 'Outstate', 'Books', 'Personal'], y_vars='Private', hue='Private', size=4, aspect=0.6)
rel3 = sns.pairplot(college_data, x_vars=['PhD', 'Terminal', 'perc_alumni', 'Grad_Rate', 'Expend', 'S_F_Ratio'], y_vars='Private', hue='Private', size=4, aspect=0.6)


# In[13]:


#Correlation between variables
college_data.corr()      

plt.figure(figsize=(15,9))
correlation_heatmap = sns.heatmap(college_data.corr(), vmin=-1, vmax=1, annot=True)
plt.title('Correlation Heatmap')


# In[114]:


#imputing binary values inplace of Yes and No
college_data.Private.replace({'Yes':1, 'No':0}, inplace=True) 


# In[115]:


print(college_data)


# ### Step 4 : Feature Selection

# In[181]:


#assigning predictor to a new variable names X1
X1 = college_data[['Top10perc', 'Top25perc', 'Room_Board', 'Outstate', 
                   'Books', 'Personal', 'PhD', 'Terminal', 'perc_alumni', 'Grad_Rate', 'Expend', 'S_F_Ratio']]

#assigning Private to a new variable named 'y1'
y1 = college_data[['Private']]


# In[182]:


from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[183]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression


# In[184]:


## Forward selection
lm = LinearRegression()
sfs1 = SFS(lm, k_features=4, forward=True, verbose=2, scoring='neg_mean_squared_error')


# In[185]:


sfs1 = sfs1.fit(X1, y1)


# In[186]:


sfs1.subsets_


# In[187]:


#backward selection
lm2 = LinearRegression()

sfs2 = SFS(lm2, k_features=4, forward=False, verbose=1, scoring='neg_mean_squared_error')
sfs2 = sfs2.fit(X1, y1)


# In[188]:


sfs2.subsets_


# In[189]:


feat_names = list(sfs2.k_feature_names_)
print(feat_names)           #feature names


# In[190]:


## Stepwise selection (bi directional)
lm3 = LinearRegression()
sfs3 = SFS(lm3, k_features=(3,4), forward=True, floating = True,verbose=2, scoring='neg_mean_squared_error')
sfs3.fit(X1, y1)


# In[191]:


sfs3.subsets_


# #### ANOVA Test

# In[192]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# In[193]:


#Model 1 (Forward selection, Stepwise selection(bi directional))
aov = ols('Private~Outstate+PhD+perc_alumni+S_F_Ratio', college_data).fit()  
anova_test = anova_lm(aov, typ = 2)                
print(anova_test)


# In[194]:


#Model 2 (Forward selection, backward)
aov2 = ols('Private~Outstate+Terminal+Expend+S_F_Ratio', college_data).fit()  
anova_test2 = anova_lm(aov2, typ = 2)                
print(anova_test2)


# In[195]:


#Model 3 (all three)
aov3 = ols('Private~Outstate+Terminal+Expend+S_F_Ratio+PhD+perc_alumni', college_data).fit()  
anova_test3 = anova_lm(aov3, typ = 2)                
print(anova_test3)


# #### AIC BIC check

# In[84]:


print(aov.aic)
print(aov2.aic)
print(aov3.aic)


# In[85]:


print(aov.bic)
print(aov2.bic)
print(aov3.bic)


# ### Step 5: Split the data into Test & Train sets and Perform Logistic regression

# In[196]:


from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
from statsmodels.formula.api import glm


# #### Logistic regression using glm library

# In[197]:


formula1 = 'Private~Outstate+Terminal+Expend+S_F_Ratio+PhD+perc_alumni'


# In[198]:


model1 = smf.glm(formula = formula1, data = college_data, family=sm.families.Binomial())
result1 = model.fit()
print(result1.summary())


# In[199]:


formula = 'Private~Outstate+Terminal+S_F_Ratio+PhD+perc_alumni'


# In[200]:


model = smf.glm(formula = formula, data = college_data, family=sm.families.Binomial())
result = model.fit()
print(result.summary())


# In[201]:


print("Coefficeients")
print(result.params)
print()
print("p-Values")
print(result.pvalues)
print()
print("Dependent variables")
print(result.model.endog_names)


# In[202]:


predictions = result.predict()
print(predictions[0:10])


# #### Logistic regression using sklearn.linear_model on training data

# In[203]:


#assigning predictor to a new variable names X2
X2 = college_data[['Outstate','Terminal','S_F_Ratio','PhD','perc_alumni']]

#assigning Private to a new variable named 'y2'
y2 = college_data[['Private']]


# In[204]:


from sklearn.model_selection  import train_test_split
X2_train, X2_test, y2_train, y2_test = 
train_test_split(X2, y2, test_size = 0.30, random_state = 0)


# In[205]:


X2_train.shape


# In[206]:


X2_train.head()


# In[207]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X2_train = sc.fit_transform(X2_train)
X2_test = sc.transform(X2_test)


# In[208]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
mod = classifier.fit(X2_train, y2_train)


# In[209]:


#Predicting the test results
y_pred2 = classifier.predict(X2_test)


# In[210]:


y_pred2      #Pritning the prediction results


# In[211]:


print('Accuracy: {:.2f}'.format(classifier.score(X2_test, y2_test)))    #verifying accuracy


# ### Step 6 : Confusion matrix

# In[96]:


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay


# In[212]:


#Confusion Matrix
cm = confusion_matrix(y2_test, y_pred2)
print(cm)


# In[213]:


#Reference : https://towardsdatascience.com/demystifying-confusion-matrix-confusion-9e82201592fd
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()    
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(y2_test, y_pred2)
np.set_printoptions(precision=2)    
    
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Positive','Negative'],
                      title='Confusion matrix, without normalization')


# ### Step 7.  Accuracy, Precision, Recall, and Specificity

# In[214]:


cl_report=classification_report(y2_test,y_pred2)
print(cl_report)


# In[221]:


#Accuracy = (TP+TN)/(TP+FP+FN+TN)
Accuracy = (51+160)/(51+11+12+160)
print('Accuracy :', Accuracy)

#Precision = TP/(TP+FP)
precision = 51/(51+11)
print('Precision :', precision)

#Recall = TP/(TP+FN)
recall = 51/(51+12)
print('Recall :', recall)

#Specificity
specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)

#F1-score (aka F-Score / F-Measure) : 2*(Recall * Precision) / (Recall + Precision)
f_Score = 2*(0.8225806451612904 * 0.8095238095238095) / (0.8225806451612904 + 0.8095238095238095)
print('F_Score : ', f_Score)



# ### Step 8 : AUC & ROC curve

# In[100]:


from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
import sklearn.metrics as metrics


# In[218]:


# calculate the fpr and tpr for all thresholds of the classification
probs = mod.predict_proba(X2_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y2_test, y_pred2)
roc_auc = metrics.auc(fpr, tpr)


# In[222]:


import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print("Accuracy", metrics.accuracy_score(y2_test, y_pred2))


# In[ ]:





# In[ ]:




