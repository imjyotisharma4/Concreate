# importing library

import os
import numpy as np
import pandas as pd
import seaborn as sns


# Increase the print output

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


os.chdir("C:\\Users\\hp\\Documents\\DaTa sci\\Project Data")

# Read the data
Raw=pd.read_csv("concrete_data.csv")
Raw.head()

#checking null values
Raw.isna().sum()
Raw.shape


# Add Intercept Column
from statsmodels.api import add_constant
Raw2 = add_constant(Raw)
Raw2.shape

# Summary
summary=Raw2.describe()
summary

# Sampling
from sklearn.model_selection import train_test_split
Train,Test= train_test_split(Raw2,test_size=0.3, random_state=17)


depVar="concrete_compressive_strength"
trainX=Train.drop(["concrete_compressive_strength"],axis=1).copy()
trainY=Train[depVar].copy()
testX=Test.drop(["concrete_compressive_strength"],axis=1).copy()
testY=Test[depVar].copy()

trainX.shape
testX.shape


# Model Building

from statsmodels.api import OLS
Model = OLS(trainY,trainX).fit()
Model.summary()


# Model Prediction

Test_Pred = Model.predict(testX)
Test_Pred[0:6]
testY[:6]



# Homoskedasticity check
sns.scatterplot(Model.fittedvalues, Model.resid) 

# Normality of errors check
sns.distplot(Model.resid) 


#########################
# Model Evaluation
#########################

# RMSE
np.sqrt(np.mean((testY - Test_Pred)**2))                                 # 9.85

# MAPE (Mean Absolute Percentage Error)
(np.mean(np.abs(((testY - Test_Pred)/testY))))*100                         #% 27

df=pd.DataFrame(data={'Predicted Values':testY,'Actual Values':Test_Pred})
df
df.to_csv("compressiveDf.csv")



# Alternate method
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(trainX,trainY)


dff=pd.DataFrame(data={'Predicted Values':testY,'Actual Values':Test_Pred})
dff
dff.to_csv("compressiveDff.csv")


###################################
# Decision Tree
###################################
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.pyplot import figure, savefig, close

from sklearn.tree import DecisionTreeRegressor
Mod = DecisionTreeRegressor(random_state = 123,min_samples_leaf = 50).fit(trainX, trainY) # Indep, Dep
testPrediction = Mod.predict(testX)

# Vizualization of DT
figure(figsize = [20, 10])
DT_Plot2 = plot_tree(Mod, fontsize = 10, feature_names = trainX.columns, 
                     filled = True)
# RMSE
np.sqrt(np.mean((testY - testPrediction)**2))                          #10.5

# MAPE
(np.mean(np.abs(((testY - testPrediction)/testY))))*100                #30%


dtf=pd.DataFrame(data={'Predicted Values':testY,'Actual Values':testPrediction})
dtf
dtf.to_csv("compressiveDT.csv")




###################################
# Random Forest
###################################
from sklearn.ensemble import RandomForestRegressor

M2 = RandomForestRegressor(random_state = 123).fit(trainX, trainY) # Indep, Dep
testPrediction = M2.predict(testX)

# RMSE
np.sqrt(np.mean((testY - testPrediction)**2))                    # 4.84

# MAPE
(np.mean(np.abs(((testY - testPrediction)/testY))))*100          # 13.0%


drf=pd.DataFrame(data={'Predicted Values':testY,'Actual Values':testPrediction})
drf
drf.to_csv("compressiveDR.csv")



##########################
# KNN
##########################
from sklearn.preprocessing import StandardScaler

Train_Scaling = StandardScaler().fit(trainX) 
trainX_Std = Train_Scaling.transform(trainX) 
testX_Std  = Train_Scaling.transform(testX) 

# Add the column names to trainX_Std, testX_Std
trainX_Std = pd.DataFrame(trainX_Std, columns = trainX.columns)
testX_Std = pd.DataFrame(testX_Std, columns = testX.columns)


from sklearn.neighbors import KNeighborsRegressor

M3 = KNeighborsRegressor().fit(trainX_Std, trainY) # Indep, Dep
testPrediction = M3.predict(testX)

print("RMSE: ", np.sqrt(np.mean((testY - testPrediction)**2)))                       #26
print("MAPE: ", (np.mean(np.abs(((testY - testPrediction)/testY))))*100)             # 54%


dknn=pd.DataFrame(data={'Predicted Values':testY,'Actual Values':testPrediction})
dknn
dknn.to_csv("compressiveknn.csv")

