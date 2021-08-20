import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plotly.express as px
from sklearn.cluster import KMeans


page_bg_img = '''
<style>
body {
background-image: url("https://github.com/kunalgupta2616/Iris-Specie-Predictor/blob/master/iris.jpg?raw=true");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)



def confusion(test_y,pred_y):
    from sklearn.metrics import confusion_matrix
    mat = confusion_matrix(test_y, pred_y)
    return mat

st.title(body='Predict Iris Species')

data=pd.read_csv(r'https://raw.githubusercontent.com/kunalgupta2616/Iris-Specie-Predictor/master/Iris.csv',
                usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
x= data.iloc[:,:-1].values
y=data.iloc[:,-1].values

show_data=st.button('Show Data')
if show_data:
    st.dataframe(data)
    show_data=st.button('Hide Data')
st.subheader("Data Information")
st.write("Number of Rows :",data.shape[0])
st.write("Number of Columns :",data.shape[1])
st.subheader("Description Report")
st.dataframe(data.describe())

from sklearn.preprocessing import LabelEncoder 
ly = LabelEncoder()
y = ly.fit_transform(y)
Model = st.sidebar.selectbox('Select Model/Algorithm',['Logistic Regression','SVM','NB Classifier','Decision Tree Classifier','KMeans Clustering'])

def hyperparameters(Model):
    params={}
    if Model=="Logistic Regression":
        params={}
        solver = st.sidebar.selectbox("Solver",['sag','saga'])
        random_state = st.sidebar.text_input("Random State",value=0)
        max_iter=st.sidebar.text_input("Maximum Iterations",value=100)
        params['solver']=solver
        params['random_state']=int(random_state)
        params['max_iter']=int(max_iter)
    elif Model=="SVM":
        params={}
        C = st.sidebar.slider("C",0.01,10.0)
        random_state = st.sidebar.text_input("Random State",value=0)
        params['C']=C
        params['random_state']=int(random_state)
    elif Model=="NB Classifier":
        params={}
        st.sidebar.text("Using Default Hyperparameters.")
    elif Model=="Decision Tree Classifier":
        params={}
        max_depth = st.sidebar.slider("Maximum Depth of Tree",min_value=1)
        random_state = st.sidebar.text_input("Random State",value=0)
        params['max_depth']=max_depth
        params['random_state']=int(random_state)
    else:
        K = st.sidebar.slider("Number of Clusters",1,10)
        params['n_clusters']=int(K)
    return params

params = hyperparameters(Model)
st.sidebar.markdown("Scroll Below in Main window to make your own predictions.")
def classifier(Model,params):
    if Model=="Logistic Regression":
        st.header('Logistic Regression')
        clf = LogisticRegression(**params)

    elif Model=="SVM":
        st.header('Support Vector Classifier')
        clf = SVC(**params)
    
    elif Model=='NB Classifier':
        st.header('Gaussian Naive Bayes Classifier')
        clf = GaussianNB()
    elif Model=="Decision Tree Classifier":
        st.header('Decision Tree Classifier')
        clf = DecisionTreeClassifier(**params)
    else:
        st.header('KMeans Clustering')
        clf = KMeans(**params)
    return clf

clf = classifier(Model,params)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

if Model =='KMeans Clustering':
    clf.fit(x)
else:
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    st.header("Evaluation Metrics")
    acc = accuracy_score(y_test,y_pred)
    st.subheader(f"Accuracy Score :")
    st.write(acc)

    prec = precision_score(y_test,y_pred,average=None)
    st.subheader(f"Precision Score : ")
    st.write(prec)

    st.subheader("Confusion Matrix :")
    mat = confusion(y_test,y_pred)
    st.write(mat)

    st.subheader("Classification Report :")
    report = classification_report(y_test, y_pred,output_dict=True,target_names=['Setosa','Versicolour','Virginica'])
    st.dataframe(report)

def visualizations(Model):
    st.header("Visualizations")
    with st.spinner("Generating visualizations..."):
        fig = px.scatter_3d(
        data, x='SepalLengthCm', y='SepalWidthCm', z='PetalWidthCm',  color='Species',symbol='Species',
        size='PetalLengthCm',title='3D-Scatter Plot of 3 features<br>Size parameter="PetalLengthCm"'
        )
    st.success("Visualizations generated successfully.")
    st.plotly_chart(fig)
    
    if Model == 'KMeans Clustering':
        scores = [KMeans(n_clusters=i+1).fit(x).inertia_ 
                for i in range(0,10)]
        no_of_clusters = np.arange(1,11)
        fig1 = px.line(x=no_of_clusters, y=scores, title="Elbow Method <br>Inertia of k-Means versus number of clusters")
        fig1.update_layout(xaxis_title_text='Number of clusters',yaxis_title_text='Sum_of_squared_distances or Inertia',width=900,height=500)
        st.plotly_chart(fig1)

if Model=='Decision Tree Classifier':
        img_url = r"https://github.com/shriyaharal7/Iris-using-Streamlit/blob/main/iris_dectree_dtreeviz.svg"
        st.image(img_url,caption="Decision Tree",use_column_width=True)

visualizations(Model)

def cust_predition(clf):
    st.header("Predict Species on Custom Data")
    seplen = st.slider('SepalLengthCm',min_value=0.1,max_value=15.0,)
    sepwid = st.slider('SepalWidthCm',0.1,20.0)
    petlen = st.slider('PetalLengthCm',0.1,20.0)
    petwid = st.slider('PetalWidthCm',0.1,20.0)
    user_data=[seplen,sepwid,petlen,petwid]
    pred = clf.predict([user_data])
    return pred

pred = cust_predition(clf)

st.write("## The Predicted Species is :",ly.inverse_transform(pred))
