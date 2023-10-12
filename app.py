from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and LabelEncoder
le = joblib.load('artifacts/label_encoder.pkl')
model = joblib.load('artifacts/random_forest_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def classify_mushroom():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        mushroom_data = pd.DataFrame([form_data])
        mushroom_data_encoded = mushroom_data.copy()
        
        try:
            for col in mushroom_data.columns:
                if mushroom_data[col].dtype == 'object':
                    mushroom_data_encoded[col] = le.fit_transform(mushroom_data[col])
        except KeyError as e:
            return render_template('index.html', result=f'Invalid input: {e}')


        prediction = model.predict(mushroom_data_encoded)[0]
        if prediction == 0:
            result = 'Poisonous'
        else:
            result = 'Edible'

        return render_template('index.html', result=result)

    return render_template('index.html', result='')

if __name__ == '__main__':
    app.run(debug=True)
