from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import io
import base64
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from warnings import filterwarnings

filterwarnings('ignore')
#to link css to app.py
app = Flask(__name__, static_url_path='/static/styles.css')

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded file and target attribute name
    file = request.files['file']
    target_attribute = request.form['target_attribute']
    target_attribute1=target_attribute
    

    # Load the dataset
    df = pd.read_csv(file)
    df1=df[target_attribute]
    unique_labels = df[target_attribute].unique()
    def convert_to_standard_format(date_str):
        formats = ['%m-%d-%Y %H:%M', '%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %I:%M:%S %p', '%m-%d-%Y %H:%M:%S']
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        return None
    # Convert the 'DATE OF OCCURRENCE' column to standard format and add as a new column
    df['Standard Dates'] = df['DATE  OF OCCURRENCE'].apply(convert_to_standard_format)
    # Convert 'DATE OF OCCURRENCE' to datetime format
    df['Standard Dates'] = pd.to_datetime(df['Standard Dates'])

    # Extract hour from 'DATE OF OCCURRENCE'
    df['hour'] = df['Standard Dates'].dt.hour
    # Group data by hour and crime type
    hour_crime_group = df.groupby(['hour', ' PRIMARY DESCRIPTION']).size().reset_index(name='Total')
    # Plotting
    fig, ax = plt.subplots(figsize=(15, 10))
    for crime_type, data in hour_crime_group.groupby(' PRIMARY DESCRIPTION'):
        ax.plot(data['hour'], data['Total'], label=crime_type, linewidth=5)
    # Customize plot
    ax.set_xlabel('Hour (24-hour clock)')
    ax.set_ylabel('Number of occurrences')
    ax.set_title('Crime Types by Hour of Day in Toronto', color='red', fontsize=25)
    ax.grid(linestyle='-')
    leg = plt.legend(fontsize='x-large')
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=4)
    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url1 = base64.b64encode(img.getvalue()).decode()

    # Plot pie chart for arrest counts
    arrest_counts = df['ARREST'].value_counts()

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(arrest_counts, labels=arrest_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Arrest Counts')
    img_arrest_counts = io.BytesIO()
    plt.savefig(img_arrest_counts, format='png')
    img_arrest_counts.seek(0)
    plot_url_arrest_counts = base64.b64encode(img_arrest_counts.getvalue()).decode()


    # Preprocessing
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    label_encoder = LabelEncoder()
    for column in non_numeric_columns:
        df[column] = label_encoder.fit_transform(df[column])
    df.fillna(df.mean(), inplace=True)
    x = df.drop([target_attribute,'Standard Dates','WARD','BEAT'], axis=1)
    
    y = [target_attribute]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, df[y], test_size=0.3, random_state=57)

    # Feature scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Initialize KNN classifier
    knn_classifier = KNeighborsClassifier()

    # Train the classifier
    knn_classifier.fit(x_train_scaled, y_train)
    

    # Predict on the testing set
    y_pred = knn_classifier.predict(x_test_scaled)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Plot bar graph
    plt.figure(figsize=(10, 6))
    df1.value_counts().plot(kind='bar', color='skyblue')
    plt.xlabel(target_attribute1)
    plt.ylabel('Count')
    plt.title('Distribution of ' + target_attribute1)
    plt.tight_layout()

    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Create a dictionary to map numerical predictions to original string labels
    #unique_labels = df[y[0]].unique()
    prediction_mapping = {i: label for i, label in enumerate(unique_labels)}

    # Convert numerical predictions to original string labels using the dictionary
    y_pred_original = [prediction_mapping[prediction] for prediction in y_pred]
    first_two_predictions = y_pred_original[:2]

    return render_template('predictions.html', predictions=first_two_predictions, accuracy=accuracy, plot_url=plot_url,plot_url1=plot_url1,
                           plot_url_bar_graph=plot_url_arrest_counts)


if __name__ == '__main__':
    app.run(debug=True)
