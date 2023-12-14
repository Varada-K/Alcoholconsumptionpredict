import pickle
from flask import Flask, request, render_template
import pandas as pd 
import numpy as np
import plotly.express as px


# Create the Flask app
app = Flask(__name__)


#viz part
df = pd.read_csv("smoking_driking_dataset_Ver01.csv")
df =df.drop_duplicates()

columns_to_check = [
    "age", "height", "weight", "waistline", "sight_left", "sight_right",
    "SBP", "DBP", "BLDS", "tot_chole", "HDL_chole", "LDL_chole", "triglyceride",
    "hemoglobin", "serum_creatinine", "SGOT_AST", "SGOT_ALT", "gamma_GTP"
]

# Calculate Z-scores for the specified columns
z_scores = (df[columns_to_check] - df[columns_to_check].mean()) / df[columns_to_check].std()

# Set the threshold for outlier detection (3σ standard deviation)
threshold = 3
outliers = np.abs(z_scores) > threshold
outlier_columns = outliers.columns[outliers.any()]

# Filter the DataFrame by removing rows containing outliers
df_cleaned = df[~outliers.any(axis=1)]
df=df_cleaned
# Sample a fraction of your dataset
sample_fraction = 0.1  
df = df.sample(frac=sample_fraction)
sample_fraction = 0.001  # Use 0.001 for 0.1%
df = df.sample(frac=sample_fraction, random_state=42)
# Get column names as a list
column_names = df.columns.tolist()




# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")  # Homepage
def homepage():
    return render_template("index.html")

@app.route("/datadescription")  # data description
def description():
    return render_template("datadescription.html")

@app.route("/dataexploration")  # exploration
def exploration_data():
    return render_template("dataexploration.html",column_names=column_names)
# def generate_graph():
#     return render_template('graph.html', column_names=column_names)

@app.route("/plot", methods=['POST'])
def plot_graph():
    selected_plot = request.form['plot_type']
    feature_x = request.form['feature_x']
    feature_y = request.form['feature_y']

    if selected_plot == 'scatter':
        fig = px.scatter(df, x=feature_x, y=feature_y, color='DRK_YN', title=f'Scatter Plot: {feature_x} vs {feature_y}')
    elif selected_plot == 'bar':
        fig = px.bar(df, x=feature_x, color='DRK_YN', title=f'Bar Plot: {feature_x} vs Drinker Status')
    elif selected_plot == 'hist':
        fig = px.histogram(df, x=feature_x, color='DRK_YN', title=f'Histogram: {feature_x} by Drinker Status')
    else:
        return 'Invalid plot type'
    return fig.to_html(full_html=False)


@app.route("/modelandinterp")  # interprete
def interp():
    return render_template("modelandinterp.html")


@app.route("/predict", methods=["POST"])  # Example prediction route
def predict():
    if request.method == "POST":
        # Get user input from the form
        input_data = [
            float(request.form['SMK_stat_type_cd']),
            float(request.form['hemoglobin']),
            float(request.form['SGOT_ALT']),
            float(request.form['age']),
            float(request.form['gamma_GTP']),
            float(request.form['tot_chole']),
            float(request.form['DBP']),
            float(request.form['SBP']),
            float(request.form['triglyceride']),
            float(request.form['BLDS']),
            float(request.form['HDL_chole'])  # Fix the typo here, remove the ']'
        ]

        # Use the model to make a prediction
        prediction = model.predict([input_data])

        # Pass the prediction to the template
        return render_template("result.html", prediction=prediction[0])


if __name__ == "__main__":
    app.run(debug=True, port=8000)