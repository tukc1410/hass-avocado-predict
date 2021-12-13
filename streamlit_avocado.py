# import libraries
# pip install pyqt5
import streamlit as st
import pandas as pd
from sklearn import metrics
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing
from math import sqrt
from fbprophet import Prophet 
from fbprophet.plot import add_changepoints_to_plot
from streamlit import caching
import matplotlib.pyplot as plt


# Source Code
data = pd.read_csv("avocado.csv")

#--------------
# GUI
st.title("DATA SCIENCE PROJECT")
st.write("# Hass Avocado Price Prediction")


# PART 2. Filter Organic Avocado - California
# Make new dataframe from original dataframe: data
df_2=data.loc[data['region']=="California"]
df_2=df_2.loc[df_2['type']=="organic"]
df_2=df_2[['Date','AveragePrice']]
df_2.index=df_2['Date']
df_2=df_2[['AveragePrice']]
df_2.sort_index(inplace=True)
df_2.index=pd.to_datetime(df_2.index)
df_2_new=df_2.reset_index()
df_2_new.columns=['ds','y']
result = seasonal_decompose(df_2, model='multiplicative')
# Train/Test Prophet
train,test=np.split(df_2_new,[int(0.7*len(df_2_new))])
# Build model
@st.cache(suppress_st_warning=True)
def load_model():
    model=Prophet() 
    model.fit(train)  
    weeks=pd.date_range('2017-04-09','2018-03-25',freq='W').strftime("%Y-%m-%d").tolist() # thời gian của test để so sánh yhat và ytest
    future=pd.DataFrame(weeks)
    future.columns=['ds']
    future['ds']=pd.to_datetime(future['ds']) 
    return model.predict(future)

forecast=load_model()
df_2_new.y.mean()
# 51 weeks in test 
test.y.mean()
y_test=test['y'].values
y_pred=forecast['yhat'].values[:51]
mae_p=mean_absolute_error(y_test,y_pred)
rmse_p=sqrt(mean_squared_error(y_test,y_pred))
# Long-term prediction for the next 1-5 years => Consider whether to expand cultivation/production, and trading
y_test_value=pd.DataFrame(y_test,index=pd.to_datetime(test['ds']),columns=['Actual'])
y_pred_value=pd.DataFrame(y_pred,index=pd.to_datetime(test['ds']),columns=['Prediction'])
m=Prophet()
m.fit(df_2_new)
def load_model2():
    #predict for next 52 weeks (1 year)         
    future_1=m.make_future_dataframe(periods=52,freq='W')
    return m.predict(future_1)
forecast_1=load_model2()
@st.cache(suppress_st_warning=True)
def load_model3():
    #predict for next 52 weeks (1 year)         
    future_2=m.make_future_dataframe(periods=260,freq='W')
    return m.predict(future_2)
forecast_2=load_model3()


# PART 3. Filter Conventional Avocado - California      
def region():
    return st.selectbox("Regions:",
                    ['','Northeast','Southeast','NorthernNewEngland','SouthCentral','RaleighGreensboro','Detroit','California','Columbus',
                    'NewYork','StLouis','HarrisburgScranton','LosAngeles','GrandRapids','Boise','Seattle','Atlanta','Chicago','Portland',
                    'RichmondNorfolk','LasVegas','Pittsburgh','Houston','SanFrancisco','NewOrleansMobile','TotalUS','HartfordSpringfield',
                    'Denver','Louisville','Boston','Indianapolis','Albany','PhoenixTucson','SanDiego','Plains','Tampa','SouthCarolina',
                    'West','Roanoke','BaltimoreWashington','Charlotte','Midsouth','Jacksonville','GreatLakes','Orlando','DallasFtWorth'])
region=region() 

def type():            
    return st.selectbox("Type:",
                                ['','organic','conventional'])
type=type()


def model():
    return st.selectbox("Model:",
                        ['','facebook prophet','holtwinters'])
model=model()
        

if (region!="" and type!="" and model=="holtwinters"):
    df_3_new=data.loc[data['region']==region]
    df_3_new=df_3_new.loc[df_3_new['type']==type] 
else:  
    df_3_new=data.loc[data['region']=="California"]
    df_3_new=df_3_new.loc[df_3_new['type']=="conventional"]

df_3_new=df_3_new[['Date','AveragePrice']]
df_3_new.index=df_3_new['Date']
df_3_new=df_3_new[['AveragePrice']]
df_3_new.sort_index(inplace=True)
df_3_new.index=pd.to_datetime(df_3_new.index)
result2 = seasonal_decompose(df_3_new, model='multiplicative')
train3,test3=np.split(df_3_new,[int(0.7*len(df_3_new))])
model3 = ExponentialSmoothing(train3, seasonal='mul', 
                             seasonal_periods=52).fit()

@st.cache(suppress_st_warning=True)
def load_model7():    
    return model3.predict(start=test3.index[0], 
                     end=test3.index[-1])
pred3=load_model7()
mae3 = mean_absolute_error(test3,pred3)
rmse3=mean_squared_error(test3,pred3)
#predict for next 52 weeks (1 year)
import datetime
@st.cache(suppress_st_warning=True)
def load_model8():   
    s = datetime.datetime(2018, 3, 25)
    e = datetime.datetime(2019, 3,  24)
    return model3  .predict(start= s, end=e)
x=load_model8()
pred_next_52_week3=x[1:]

next_52_week3=x.index[1:]
values_next_52_week3=x.values[1:]
#predict for next 260 weeks (5 years)
@st.cache(suppress_st_warning=True)
def load_model9():  
    s2 = datetime.datetime(2018, 3, 25)
    e2 = datetime.datetime(2023, 3,  19)
    return model3.predict(start= s2, end=e2)
x2=load_model9()
pred_next_260_week4=x2[1:]
next_260_week4=x2.index[1:]
values_next_260_week4=x2.values[1:] 


# PART 4. Organic avocado in SanDiego
# Make new dataframe from original dataframe: data

if (region!="" and type!="" and model=="facebook prophet"):
    df_4=data.loc[data['region']==region]
    df_4=df_4.loc[df_4['type']==type]
    
else:   
    df_4=data.loc[data['region']=="SanDiego"]
    df_4=df_4.loc[df_4['type']=="organic"]
df_4=df_4[['Date','AveragePrice']]
df_4.index=df_4['Date']
df_4=df_4[['AveragePrice']]   
df_4.sort_index(inplace=True)
df_4.index=pd.to_datetime(df_4.index)
df_4_new=df_4.reset_index()
df_4_new.columns=['ds','y']
result4 = seasonal_decompose(df_4, model='multiplicative')
    # Train/Test Prophet
train4,test4=np.split(df_4_new,[int(0.7*len(df_4_new))])
    # Build model
@st.cache(suppress_st_warning=True)
def load_model4():
    model4=Prophet() 
    model4.fit(train4)  
    weeks4=pd.date_range('2017-04-09','2018-03-25',freq='W').strftime("%Y-%m-%d").tolist() # thời gian của test để so sánh yhat và ytest
    future4=pd.DataFrame(weeks4)
    future4.columns=['ds']
    future4['ds']=pd.to_datetime(future4['ds']) 
    return model4.predict(future4)
forecast4=load_model4()
df_4_new.y.mean()
    # 51 weeks in test 
test4.y.mean()
y_test4=test4['y'].values
y_pred4=forecast4['yhat'].values[:51]
mae_p4=mean_absolute_error(y_test4,y_pred4)
rmse_p4=sqrt(mean_squared_error(y_test4,y_pred4))
    # Long-term prediction for the next 1-5 years => Consider whether to expand cultivation/production, and trading
y_test_value4=pd.DataFrame(y_test4,index=pd.to_datetime(test4['ds']),columns=['Actual'])
y_pred_value4=pd.DataFrame(y_pred4,index=pd.to_datetime(test4['ds']),columns=['Prediction'])
    #predict for next 52 weeks (1 year)
m5=Prophet()
m5.fit(df_4_new)
@st.cache(suppress_st_warning=True)
def load_model5():
    future_14=m5.make_future_dataframe(periods=52,freq='W')
    return m5.predict(future_14)
forecast_14=load_model5()
#predict for next 260 weeks (5 years)
@st.cache(suppress_st_warning=True)
def load_model6():
    future_24=m5.make_future_dataframe(periods=260,freq='W')
    return m5.predict(future_24)
forecast_24=load_model6()

if st.button("Clear history cache"):
    st.legacy_caching.clear_cache()

if st.button("Submit"):
    st.legacy_caching.clear_cache()
    
# GUI
menu = ["Business Objective", "Content"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.header("Business Objective")
    st.write("""
    ##### Bơ “Hass”, một công ty có trụ sở tại Mexico, chuyên sản xuất nhiều loại quả bơ được bán ở Mỹ. Họ đã rất thành công trong những năm gần đây và muốn mở rộng. Vì vậy, họ muốn xây dựng mô hình hợp lý để dự đoán giá trung bình của bơ “Hass” ở Mỹ nhằm xem xét việc mở rộng các loại trang trại Bơ đang có cho việc trồng bơ ở các vùng khác.
    """)  
    st.write("""
    ##### => Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ => xem xét việc mở rộng sản xuất, kinh doanh.
    """)
    st.image("image4.jpg")
    st.subheader("Content")
    st.write("""
    #### Part 1: Overview Dataset
    """)
    st.write("""
    #### Part 2: Organic Avocado Average Price Prediction in California 
    """)
    st.write("""
    #### Part 3: Conventional Avocado Average Price Prediction in California 
    """)
    st.write("""
    #### Part 4: Organic Avocado Average Price Prediction in SanDiego
    """)
    st.write("""
     
    """)
    col1, col2, col3 = st.columns([1,6,1])
     
    with col1:
        st.write("")

    with col2:
        st.image(["image.jpg","image5.jpg"])

    with col3:
        st.write("")
    
elif choice=="Content":
    menu2 = ["Overview Dataset","Organic Avocado-California", "Conventional Avocado-California", "Organic Avocado-SanDiego","New Prediction"]
    choice = st.sidebar.selectbox('Menu of Content', menu2)
    if choice=="Overview Dataset": 
        st.header("Part 1. Overview Dataset")  
        st.write(""" #### 1.1. Organic's Average Price more expensive than Conventional's. Average Price is effected by Type
        """)
        fig1, ax = plt.subplots(figsize=(20,8))
        sns.boxplot(data=data,x="type",y="AveragePrice")
        plt.show()
        st.pyplot(fig1) 
    
        st.write(""" #### 1.2. Organic's Average Price is effected by Region
        """)
        fig2,ax=plt.subplots(figsize=(22,12))
        sns.boxplot(data=data[data['type']=='organic'],
            x="region",y="AveragePrice",ax=ax)
        plt.xticks(rotation=90)
        plt.show()
        st.pyplot(fig2)

        st.write(""" #### 1.3. Conventional's Average Price is effected by Region
        """)
        fig4,ax=plt.subplots(figsize=(22,12))
        sns.boxplot(data=data[data['type']=='conventional'],
            x="region",y="AveragePrice",ax=ax)
        plt.xticks(rotation=90)
        plt.show()
        st.pyplot(fig4)  

        st.write(""" #### 1.4. Total volume has high correlation with 4046, 4225, 4770, Total Bags, Small Bags, Large Bags)
        """)
        corr=data.corr()
        fig3,ax=plt.subplots(figsize=(10,10))
        sns.heatmap(corr,vmin=-1,vmax=1,annot=True)
        plt.show()
        st.pyplot(fig3) 

    elif choice == 'Organic Avocado-California':            
        menu3 = ["Overview","Build Project","Show Prediction"]
        choice = st.sidebar.selectbox('Menu of Organic Avocado - California', menu3)
        st.write("""### Part 2: Organic Avocado Average Price Prediction in California 
        
        """)
        if choice=="Overview":           
            st.write("""
            #### 2.1. Overview organic avocado in California
            ##### FACEBOOK PROPHET - Time Series Algorithm
            """)

            fig5,ax=plt.subplots(figsize=(8,8))
            plt.plot(df_2)
            plt.title("AveragePrice-Organic",color='red',fontsize=20)
            plt.show();
            st.pyplot(fig5)

            
            fig6,ax=plt.subplots(figsize=(9,3))
            result.seasonal.plot()
            plt.show()
            st.pyplot(fig6)

            fig7,ax=plt.subplots(figsize=(9,3))
            result.trend.plot()
            plt.show()
            st.pyplot(fig7)

            st.write("### Organic avocado average price is seasonal and has a significant increasing trend, highest in September and lowest in March.")
            st.write("### Results: ")
            st.write("""
            #### FaceBook Prophet : MAE- 0.151
            """)
            st.write(""" 
            #### Arima            : MAE- 0.138
            """)
            st.write("""
            #### Holtwinters      : MAE- 0.163
            """)
        elif choice == 'Build Project':
            st.subheader("Build Project - FaceBook Prophet")
            st.write("""
            ##### Some data:
            """)
            st.dataframe(df_2_new.head(3))
            st.dataframe(df_2_new.tail(3))   
            st.text("Mean of Organic Avocado AveragePrice in California: " + str(round(df_2_new['y'].mean(),2)) + " USD")
            st.write("""
            ##### Build model ...
            """)
            st.write("""
            #### Calculate MAE between expected and predicted values
            """)
            st.write("MAE: " + str(round(mae_p,2)))
            
            st.write("""This result shows that Prophet's MAE are good enough to predict the organic avocado AveragePrice in California, MAE = 0.15 (about 10% of the AveragePrice), compared to the AveragePrice ~ 1.69.
                """)
            st.write("##### Visualization: AveragePrice vs AveragePrice Prediction from 04-2017 to 03-2018 (51 weeks)")
                # Visulaize the result
            fig, ax = plt.subplots()   
            plt.plot(y_test_value,label='Real AveragePrice')
            plt.plot(y_pred_value,label='Prediction AveragePrice')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()   
            st.pyplot(fig)  

        elif choice == 'Show Prediction':
            st.subheader(" Prediction for the future in California - Organic Avocado - Facebook Prophet")
            st.write("##### Next 52 weeks-1 year")
   
                # Next 1 years   
            fig1=m.plot(forecast_1)
            fig1.show()
            a=add_changepoints_to_plot(fig1.gca(),m,forecast_1)
            st.pyplot(fig1)

            fig2, ax = plt.subplots()     
            plt.plot(df_2_new['y'],label='AveragePrice')
            plt.plot(forecast_1['yhat'],label='AveragePrice_Prediction',color='red')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()
            st.pyplot(fig2)
    
            st.write("##### Next 260 weeks-5 years")
            # Next 5 years   
            fig3=m.plot(forecast_2)
            fig3.show()
            a=add_changepoints_to_plot(fig3.gca(),m,forecast_2)
            st.pyplot(fig3)

            fig4, ax = plt.subplots()  
            plt.plot(df_2_new['y'],label='AveragePrice')
            plt.plot(forecast_2['yhat'],label='AveragePrice_Prediction',color='red')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()
            st.pyplot(fig4)
            st.write(""" ### Conclusion: The average price of organic avocado in California tends to increase in short-term (52 weeks-1year) and long-term(260 weeks-5years).Company should consider business expansion of organic avocado in California.
                """)       

    elif choice == 'Conventional Avocado-California':
        st.write("""
        ### Part 3: Conventional Avocado Average Price Prediction in California 
        ##### HOLTWINTERS - Time Series Algorithm
        """)
        menu4 = ["Overview","Build Project","Show Prediction"]
        choice = st.sidebar.selectbox('Menu of Conventional Avocado - California', menu4)
        if choice=="Overview":
            st.write("""
            #### 2.1. Overview conventional avocado in California
            """)
            fig7,ax=plt.subplots(figsize=(8,8))
            plt.plot(df_3_new)
            plt.title("AveragePrice-Conventional",color='red',fontsize=20)
            plt.show();
            st.pyplot(fig7)

            
            fig8,ax=plt.subplots(figsize=(9,3))
            result2.seasonal.plot()
            plt.show()
            st.pyplot(fig8)

            fig9,ax=plt.subplots(figsize=(9,3))
            result2.trend.plot()
            plt.show()
            st.pyplot(fig9)

            st.write("### Conventional avocado average price is seasonal and has a significant increasing trend, highest in September and lowest in March.")
            st.write("### Results: ")
            st.write("""
            #### FaceBook Prophet : MAE- 0.170
            """) 
            st.write("""
            #### Arima            : MAE- 0.184
            """)
            st.write("""
            #### Holtwinters      : MAE- 0.152
            """)
        elif choice == 'Build Project':
            st.subheader("Build Project - Holtwinters")
            st.write("""
            ##### Some data:
            """)
            st.dataframe(df_3_new.head(3))
            st.dataframe(df_3_new.tail(3))   
            st.text("Mean of Conventional Avocado AveragePrice in California: " + str(round(df_3_new['AveragePrice'].mean(),2)) + " USD")
            st.write("""
            ##### Build model ...
            """)
            st.write("""
            #### Calculate MAE between expected and predicted values
            """)
            st.write("MAE: " + str(round(mae3,2)))
           
            st.write("""This result shows that MAE are good enough to predict the organic avocado AveragePrice in California, MAE = 0.15 (about 13% of the AveragePrice), compared to the AveragePrice ~ 1.1.
                """)
            st.write("##### Visualization: AveragePrice vs AveragePrice Prediction from 04-2017 to 03-2018 (51 weeks)")
                # Visulaize the result

            fig10,ax=plt.subplots()
            plt.plot(test3, label='AveragePrice')
            plt.plot(pred3, label='Prediction')
            plt.xticks(rotation='vertical') 
            plt.legend()             
            plt.show()
            st.pyplot(fig10)

        elif choice == 'Show Prediction':
            st.subheader(" Prediction for the future in California - Conventional Avocado - Holtwinters")
            st.write("##### Next 52 weeks-1 year")
   
            fig11,ax=plt.subplots(figsize=(10,6))
            plt.title("AveragePrice from 2015-01 to 2018-03 and next 52 weeks")
            plt.plot(train3.index, train3, label='Train')
            plt.plot(test3.index, test3, label='Test')
            plt.plot(pred3.index, pred3, label='Predict')
            plt.plot(x.index, x.values, label='Next-52-weeks')
            plt.legend(loc='best')            
            st.pyplot(fig11)

            
            st.write("##### Next 260 weeks-5 years")
            fig12,ax=plt.subplots(figsize=(10,6))
            plt.title("AveragePrice from 2015-01 to 2018-03 and next 260 weeks")
            plt.plot(train3.index, train3, label='Train')
            plt.plot(test3.index, test3, label='Test')
            plt.plot(pred3.index, pred3, label='Predict')
            plt.plot(x2.index, x2.values, label='Next-260-weeks')
            plt.legend(loc='best')
            st.pyplot(fig12)
            st.write(""" ### Conclusion: The average price of convention avocado in California tends to not increase in short-term (52 weeks-1year) and long-term(260 weeks-5years).Company should not expand conventional avocado's business in California.
                """)   

    elif choice == 'Organic Avocado-SanDiego':
                  
        menu6 = ["Overview","Build Project","Show Prediction"]
        choice = st.sidebar.selectbox('Menu of Organic Avocado - SanDiego', menu6)
        st.write("""### Part 2: Organic Avocado Average Price Prediction in SanDiego
        
        """)
        if choice=="Overview":           
            st.write("""
            #### 2.1. Overview 
            ##### FACEBOOK PROPHET - Time Series Algorithm
            """)

            fig13,ax=plt.subplots(figsize=(8,8))
            plt.plot(df_4)
            plt.title("AveragePrice-Organic-SanDiego",color='red',fontsize=20)
            plt.show();
            st.pyplot(fig13)

            
            fig14,ax=plt.subplots(figsize=(9,3))
            result4.seasonal.plot()
            plt.show()
            st.pyplot(fig14)

            fig15,ax=plt.subplots(figsize=(9,3))
            result4.trend.plot()
            plt.show()
            st.pyplot(fig15)
            st.write("### Organic avocado average price is seasonal and has a significant increasing trend, highest in September and lowest in March.")
            st.write("### Results: ")
            st.write("""
            #### FaceBook Prophet : MAE- 0.214
            """)
            st.write(""" 
            #### Arima            : MAE- 0.195
            """)
            st.write("""
             #### Holtwinters      : MAE- 0.224
            """)
        elif choice == 'Build Project':
            st.subheader("Build Project - FaceBook Prophet")
            st.write("""
            ##### Some data:
            """)
            st.dataframe(df_4_new.head(3))
            st.dataframe(df_4_new.tail(3))           
            st.text("Mean of Organic Avocado AveragePrice in SanDiego: " + str(round(df_4_new['y'].mean(),2))+ "USD")
            
            st.write("""
            ##### Build model ...
            """)
            st.write("""
            #### Calculate MAE between expected and predicted values
            """)
            st.write("MAE: " + str(round(mae_p4,2)))
            st.write("""This result shows that Prophet's MAE are good enough to predict the organic avocado AveragePrice in SanDiego, MAE = 0.21 (about 12% of the AveragePrice), compared to the AveragePrice ~ 1.73.
                """)
            st.write("##### Visualization: AveragePrice vs AveragePrice Prediction from 04-2017 to 03-2018 (51 weeks)")
                # Visulaize the result
            fig16, ax = plt.subplots()   
            plt.plot(y_test_value4,label='Real')
            plt.plot(y_pred_value4,label='Prediction')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()   
            st.pyplot(fig16)  

        elif choice == 'Show Prediction':
            st.subheader(" Prediction for the future in SanDiego - Organic Avocado - Facebook Prophet")
            st.write("##### Next 52 weeks-1 year")
            

            # Next 1 years   
            fig17=m.plot(forecast_14)
            fig17.show()
            a=add_changepoints_to_plot(fig17.gca(),m5,forecast_14)
            st.pyplot(fig17)

            fig18, ax = plt.subplots()
            plt.plot(df_4_new['y'],label='AveragePrice')
            plt.plot(forecast_14['yhat'],label='Prediction',color='red')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()
            st.pyplot(fig18)
    
            st.write("##### Next 260 weeks-5 years")
            # Next 5 years   
            fig19=m.plot(forecast_24)
            fig19.show()
            a=add_changepoints_to_plot(fig19.gca(),m5,forecast_24)
            st.pyplot(fig19)

            fig20, ax = plt.subplots()
            plt.plot(df_4_new['y'],label='AveragePrice')
            plt.plot(forecast_24['yhat'],label='Prediction',color='red')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()
            st.pyplot(fig20)
            st.write(""" ### Conclusion: The average price of organic avocado in SanDiego tends to increase in short-term (52 weeks-1year) and long-term(260 weeks-5years).Company should consider business expansion of organic avocado in SanDiego.
                """)
    elif choice=="New Prediction":
      

        if (region!="" and type!="" and model=="holtwinters"):
            st.subheader("Overview "+type +" avocado in " +region )
                
            fig7,ax=plt.subplots(figsize=(8,8))
            plt.plot(df_3_new)
            plt.title("AveragePrice-"+type+"-"+region,color='red',fontsize=20)
            plt.show();
            st.pyplot(fig7)

            
            fig8,ax=plt.subplots(figsize=(9,3))
            result2.seasonal.plot()
            plt.show()
            st.pyplot(fig8)

            fig9,ax=plt.subplots(figsize=(9,3))
            result2.trend.plot()
            plt.show()
            st.pyplot(fig9)

            st.subheader("Mean of " +type+ " Avocado AveragePrice-"+ region+": " + str(round(df_3_new['AveragePrice'].mean(),2)) + " USD")
            st.subheader("MAE: " + str(round(mae3,2)))
            st.write("##### Visualization: AveragePrice vs AveragePrice Prediction from 04-2017 to 03-2018 (51 weeks)")
            # Visulaize the result

            fig10,ax=plt.subplots()
            plt.plot(test3, label='AveragePrice')
            plt.plot(pred3, label='Prediction')
            plt.xticks(rotation='vertical') 
            plt.legend()             
            plt.show()
            st.pyplot(fig10)

            st.write("##### Next 52 weeks-1 year")
            
            fig11,ax=plt.subplots(figsize=(10,6))
            plt.title("AveragePrice from 2015-01 to 2018-03 and next 52 weeks")
            plt.plot(train3.index, train3, label='Train')
            plt.plot(test3.index, test3, label='Test')
            plt.plot(pred3.index, pred3, label='Predict')
            plt.plot(x.index, x.values, label='Next-52-weeks')
            plt.legend(loc='best')            
            st.pyplot(fig11)


            st.write("##### Next 260 weeks-5 years")
            fig12,ax=plt.subplots(figsize=(10,6))
            plt.title("AveragePrice from 2015-01 to 2018-03 and next 260 weeks")
            plt.plot(train3.index, train3, label='Train')
            plt.plot(test3.index, test3, label='Test')
            plt.plot(pred3.index, pred3, label='Predict')
            plt.plot(x2.index, x2.values, label='Next-260-weeks')
            plt.legend(loc='best')
            st.pyplot(fig12)
            
        elif (region!="" and type!="" and model=="facebook prophet"):
            st.subheader("Overview "+type +" avocado in " +region )
            fig30,ax=plt.subplots(figsize=(8,8))
            plt.plot(df_4)
            plt.title("AveragePrice-"+type+"-"+region,color='red',fontsize=20)
            plt.show();
            st.pyplot(fig30)

            
            fig31,ax=plt.subplots(figsize=(9,3))
            result4.seasonal.plot()
            plt.show()
            st.pyplot(fig31)

            fig32,ax=plt.subplots(figsize=(9,3))
            result4.trend.plot()
            plt.show()
            st.pyplot(fig32)
            st.subheader("Mean of " +type+ " Avocado AveragePrice-"+ region+": "  + str(round(df_4_new['y'].mean(),2))+ "USD")
            st.subheader("MAE: " + str(round(mae_p4,2)))


            st.write("##### Next 52 weeks-1 year")         
            # Next 1 years   
            fig33=m.plot(forecast_14)
            fig33.show()
            a=add_changepoints_to_plot(fig33.gca(),m5,forecast_14)
            st.pyplot(fig33)

            fig34, ax = plt.subplots()
            plt.plot(df_4_new['y'],label='AveragePrice')
            plt.plot(forecast_14['yhat'],label='Prediction',color='red')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()
            st.pyplot(fig34)

            st.write("##### Next 260 weeks-5 years")
            # Next 5 years   
            fig35=m.plot(forecast_24)
            fig35.show()
            a=add_changepoints_to_plot(fig35.gca(),m5,forecast_24)
            st.pyplot(fig35)

            fig36, ax = plt.subplots()
            plt.plot(df_4_new['y'],label='AveragePrice')
            plt.plot(forecast_24['yhat'],label='Prediction',color='red')
            plt.xticks(rotation='vertical')
            plt.legend()
            plt.show()
            st.pyplot(fig36)
              