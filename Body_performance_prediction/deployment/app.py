from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)

CLASS = ['D', 'C', 'B', 'A']
columns = ["age", "gender", "height" ,"weight", "body_fat","diastolic","systolic" ,"grip_force","sit_and_bend_forward" ,"sit_ups_counts","broad_jump"]
with open("body_performance.pkl", "rb") as f:
    model_body = pickle.load(f)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/body", methods=['GET', 'POST'])
def body_inference():
    if request.method == 'POST':
        data = request.json
        new_data =[data["age"],
                    data["gender"],
                    data["height"],
                    data["weight"], 
                    data["body_fat"],
                    data["diastolic"],
                    data["systolic"],
                    data["grip_force"],
                    data["sit_and_bend_forward"],
                    data["sit_ups_counts"],
                    data["broad_jump"]]

        new_data = pd.DataFrame([new_data], columns = columns)
        res = model_body.predict(new_data)

        response = {'code':200, 'status':'OK',
                    'result':{'prediction': str(res[0]),
                              'classes': CLASS[res[0]]}}
        # if CLASS[res[0]] == 'A':
        #     print('You are the best')
        # elif CLASS[res[0]] == 'B':
        #     print('You can improve')
        # elif CLASS[res[0]] == 'C':
        #     print('Hey, atleast you are healty')
        # else :
        #     print('At least we are alive, am i right??')
        
        return jsonify(response)
    return "Silahkan gunakan method post untuk mengakses model body performance"

# app.run(debug=True)
