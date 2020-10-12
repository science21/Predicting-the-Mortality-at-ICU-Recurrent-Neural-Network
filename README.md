# Predicting-the-Mortality-at-ICU-Recurrent-Neural-Network

We developed Recurrent Neural Network (RNN) models to predict mortality for patients in Intensive Care Unit (ICU). 

After cleaning the health data from MIMIC-III database through extract- transform-load (ETL) approach with Python and Apache Spark, we constructed the RNN models using PyTorch, a Python library for deep learning, and further run the model on Amazon Web Service (AWS) SageMaker. 

Our results showed that the model had a good performance to predict the mortality at ICU, with the accuracy of 81% approximately and the Area Under the Curve (AUC) of Receiver Operating Characteristics (ROC) of 88%, which is substantially improved compared with the performance of the previous baseline model. 

Data preprocessing and batch size are two major factors influencing our model performance. 

Additionally, we applied the improved model to predict the mortality of patients who stayed in ICU in less than 24 hours and less than 48 hours, respectively. 

The model had an even better performance for these types of patients, with the accuracy of 85% (AUC = 92%) and 83% (AUC = 88%), respectively. 

These findings consolidate that our model is reliable for predicting mortality of patients in ICU.

Key words: PyTorch, AWS SageMaker, Apache Spark, ICU, RNN
