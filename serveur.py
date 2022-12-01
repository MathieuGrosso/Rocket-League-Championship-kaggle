
from flask import Flask, jsonify, request
import return_predictions
import train_model from 'src/app.py'
from joblib import dump, load

app = Flask(__name__)

MODEL_FILEPATH = "/Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/model/model_A_0.joblib"


@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"


@app.route("/predict")
def predict():
    return "Welcome to machine learning model APIs!"


@app.route('/train', methods=['GET'])
try:
            train_model(MODEL_FILEPATH)
            return jsonify(({'status': 'success', 'message': 'Model successfully updated'}))

        except Exception as e:
            return str(e)

@app.route('/predict', methods=['POST'])
def get_scores(MODEL_FILEPATH="/Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/model/model_A_0.joblib"):
    payload = request.json # get the data
    print(payload)
    input_df = pd.DataFrame(payload)
    print(input_df)
    
    payload = request.json
    input_df = pd.DataFrame(payload)
    input_df.fillna(-1, inplace=True)

    # if not os.path.exists(MODEL_FILEPATH):
    #     train_and_save_model(MODEL_FILEPATH)

    model = load(MODEL_FILEPATH)
    predictions = model.predict_proba(input_df)
    scores = [prediction[1] for prediction in predictions]

    return jsonify({'scores': scores})


if __name__ == '__main__':
    app.run()

