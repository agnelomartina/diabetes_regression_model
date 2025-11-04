The model used for the diabetes dataset is linear regression and decisiontree regressor since it is continuous and has linear relationship with the datasets.
The model is trained and tested and the prediction is determined
The metrics used for this model is r2score,mean absolute error and mean square error since using a decision tree regressor
The actual value after the test data i.e(Y_test) is compared with the predicted data using Boxplot for greater visualization
the BMI data of the patients and the target data are visualised using histogram
And finally cross validated.

To achieve API using Flask we need a base file where we generate a pickle file.
With the help of pickle file an application file is setup where we use JSON where the input is recieved from the user 
and the predicted data is retrieved through JSON request and response from apicalling file.
