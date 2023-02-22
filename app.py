import pandas as pd  
import numpy as np 
import gradio as gr  

df = pd.read_csv('loan.csv') 

df.drop(['Loan_ID', 'Dependents'], axis=1, inplace=True ) 

Gender = df['Gender'].mode().sum() 
Married = df['Married'].mode().sum()  
Self_Employed = df['Self_Employed'].mode().sum() 
LoanAmount = df['LoanAmount'].median() 
Loan_Amount_Term = df['Loan_Amount_Term'].median()  
Credit_History = df['Credit_History'].median()  

df['Gender'].fillna((Gender), inplace = True) 
df['Married'].fillna((Married), inplace = True)  
df['Self_Employed'].fillna((Self_Employed), inplace = True) 
df['LoanAmount'].fillna((LoanAmount), inplace = True)  
df['Loan_Amount_Term'].fillna((Loan_Amount_Term), inplace = True) 
df['Credit_History'].fillna((Credit_History), inplace = True)  

df['Gender'].replace({'Male': '0', 'Female':'1'}, inplace = True) # Male = 0,,,Female = 1
df['Married'].replace({'No':'0', 'Yes':'1'},inplace = True) # No = 0,,,Yes = 1   
df['Education'].replace({'Graduate':'0','Not Graduate': '1' }, inplace = True) # Graduate = 0,,,Not Graduate = 1 
df['Self_Employed'].replace({'No': '0', 'Yes':'1'}, inplace = True) # No = 0,,,Yes = 1 
df['Property_Area'].replace({'Urban':'1', 'Rural':'2', 'Semiurban':'3'}, inplace=True) # Urban = 1,, Rural = 2,, Semiurban = 3  
df['Loan_Status'].replace({'N': '0', 'Y':'1'}, inplace = True) # N = 0,,,Y = 1  

from sklearn.model_selection import train_test_split  
X = df.drop("Loan_Status", axis=1) 
y = df["Loan_Status"] 
X_train, X_test, y_train, y_test = train_test_split(X,y)  

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(X_test)  

from sklearn.linear_model import LogisticRegression
model =  LogisticRegression()
model.fit(x_train_scaled, y_train) 

def loan(Gender,Married,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,
    LoanAmount,Loan_Amount_Term,Credit_History,Property_Area): 
    X = np.array([Gender,Married,Education,Self_Employed,ApplicantIncome,
    CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area])  
    pre = model.predict(X.reshape(1, -1))
    if pre == '1':
        return "Eligible for loan" 
    return "Not eligible for loan" 

app = gr.Interface(fn=loan, inputs=['number','number','number','number','number','number','number','number','number','number'], outputs='text') 
app.launch(share=True)