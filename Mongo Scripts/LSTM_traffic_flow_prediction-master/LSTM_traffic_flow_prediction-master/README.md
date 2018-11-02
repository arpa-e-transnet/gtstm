# LSTM_traffic_flow_prediction
Predict congestion based on upstream and downstream data- includes data generation files and LSTM model files

# Introduction
The goal of the project is to use historical traffic data and use various data mining and machine learning techniques to forecast the average time it will take for a user to reach from one point to another and to detect flow labels that determine the type of congestion. User Input: The user input is either the detector id from which the location could be automatically detected. It could also be a latitude and longitude location of the originating and destination point. The origin could be users current location.

# Data Attributes
The data attributes that are fed into the LSTM model include: Detector Id, start(Timestamp), duration(Interval for which the vehicles are recorded), volume, occupancy, speed, percentage of trucks, valid data. Since the model takes into the spatio temporal attributes of the travel, the spatial attributes are incorporated using attributes such as Detector Id, latitude, longitude, elevation and direction.

# Data Preprocessing
First we create a device id and a detector id mapping – this maintains the most recent version of the device.We maintain the device – longitude, latitude, direction mapping.We create a method that takes a device id, its corresponding mile marker. We then find all the devices in the same primary road with the same direction.Now we sort the resulting devices by the mile marker and obtain the devices with the downstream mile markers. For all the neighboring devices, choose all the entries from the data file and filter the entries for the current row timestamp. With this preprocessed current device and neighboring device data, we obtain the physical distance using the latitude, longitude device mapping. We also take the time difference into consideration for the current row and the neighboring device rows. For the current row, we choose an entry for the same device, detector combination fifteen minutes into the future – this will be our train y value.

# Literature Survey
I. ‘A Deep Learning Approach to the Citywide Traffic Accident Risk Prediction’ makes use of the following LSTM architecture with very similar spatial and temporal preprocessing of data'

The above mentioned paper uses a very similar approach to ours for traffic accident prediction. The first input is the sequence of recent traffic accident frequency, and it is input to the first LSTM layer. The second input contains the longitude and latitude of the region center that the model is expected to predict. The latitude longitude information is directly input into fully connected layers.

Here is the architecture diagram for the accident prediction lstm model:
![alt text](https://github.gatech.edu/CEE-Transportation/LSTM_traffic_flow_prediction/blob/master/lstm_img.png)

# Reasons for choosing LSTM

LSTM addresses the vanishing gradient problem – the network forgets the patterns that happened before a longer interval of time due to gradient descent and backpropagation through each layer of the network. “LSTM can capture the periodic feature of traffic accident, and traditional RNNs shows poor performance and intrinsic difficulties in training when it has long time period. These weaknesses have been proved in researches related with traffic flow prediction[10]. On another hand, the explicit memory cell in LSTM can avoid the problems of gradient vanish or gradient explosion existed in traditional RNNs.” – quoted by ‘A Deep Learning Approach to the Citywide Traffic Accident Risk Prediction’. 

# Using the trained LSTM model for a python application

Once the model training is complete on all of the historic data, the model can be stored as a pickle file. This pickled model can be used to predict any type of unseen data when the data is fed into it in the required format which is:

Current device id attributes for current timestamp, Current device id attributes for timestamps before/after current time, Nearby device id attributes for current timestamp, Nearby device id attributes for timestamps before/after current time.

The model will then predict the future speed with an RMSE of about 70%.

# Model performances

The model accomadates spatio temporal attributes. We have tried and tested the model for various data scenarios and the results are listed below. The following results are for ‘Adam optimizer’ and 5 epochs – Accuracy can be improved by adding more LSTM layers or by making LSTM stateful with memory.

Results:
RMSE for 13 downstream stations                       Train Score: 23.18 RMSE   Test Score: 28.05 RMSE
RMSE for 23 downstream and upstream stations          Train Score: 27.11 RMSE   Test Score: 27.03 RMSE
RMSE for 23 downstream stations                       Train Score: 48.74 RMSE   Test Score: 49.58 RMSE
RMSE for 20 downstream stations – using only location Train Score: 49.12 RMSE   Test Score: 50.71 RMSE

The data plots for the corresponding model predictions are available in the PPT.
We can see that the model works best when we take into account both the upstream and downstream devices. This is because the congestion in the the downstream station is eventually propagated to the current station. Similarly, once a congestion is cleared, the traffic flow that was accumulated during the congestion in the upstream still causes a delay in the current station. Hence it is justifiable that the model that accounts for both spatio temporal representation of the upstream and downstream devices performs accurately in predicting the future speed.

# Future Work

The model performance can be improved by adopting the following tricks:
Trying other optimizers and loss functions(Hyperparameter optimization)
Stacked LSTM with stateful memory
Increasing the number of hidden layers and lstm layers
Increasing the number of epochs

  

















