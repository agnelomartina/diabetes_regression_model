from flask import Flask,request,jsonify
import pickle
import pandas as pd

with open("diabetes_linear_trained_model.pkl","rb") as file_reading_obj:
    train_model=pickle.load(file_reading_obj)


app_name=Flask(__name__) #__name__ is an attribute or magic method

@app_name.route("/linear_model_predict",methods=["POST"])
def linear_prediction():
    data=request.get_json()#getting the data from request in the form of JSON
    bmi_from_user=data.get('my_input')
    print(bmi_from_user)
    print("##################################")
#checking if the user input is valid or not and checks if the key is appropriate from the dictionary by means of list
    if not bmi_from_user or not isinstance(bmi_from_user,list):
     return jsonify({'error msg':'please validate your input'}),400


    data_input = pd.DataFrame({"bmi":bmi_from_user})
    print(data_input)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    predicted_output=train_model.predict(data_input[["bmi"]])
    print(predicted_output)
    return jsonify({'data': predicted_output.tolist()})
    #except Exception as e:
    #return jsonify({'error': str(e)}), 500

@app_name.route("/")
def landing():
    return "welcome to Uptor"

@app_name.route("/login")
def login():
    return "welcome to my login"

#since python runs the programme in a sequential order we have to ask python to start from main
if __name__ =="__main__":
    app_name.run(debug=True)