from flask import Flask, render_template, request,flash,session
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string
import re
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from twilio.rest import Client
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask_mysqldb import MySQL
from flask_session import Session
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
Session(app)
# Load your dataset (replace 'your_data.csv' with your dataset)
data = pd.read_csv('D:\Minor Project\output_data2.csv')
data2 = pd.read_excel('D:\Minor Project\Book2.xlsx')
min_description_length = 10  # Minimum length to keep
max_description_length = 500  # Maximum length to keep
data2 = data2[data2['Business Description'].apply(len).between(min_description_length, max_description_length)]

# Select the features and labels
features = data[['Ask(in Lakhs)', 'Equity', 'Valuation Requested', 'train and test']]
labels = data['Accepted Offer']

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return ' '.join(lemmatized_sentence)

def remove_noise(data, stop_words=()):
    cleaned_tokens = []
    for token, tag in pos_tag(data):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token.lower() not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

data2['tokenized'] = data2['Business Description'].apply(lambda x: word_tokenize(x))
data2['lemmatized'] = data2['tokenized'].apply(lemmatize_sentence)
data2['cleaned_tokens'] = data2['tokenized'].apply(lambda x: remove_noise(x, stopwords.words('english')))

# Remove outliers based on TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(data2['Business Description'])

# Apply Isolation Forest for outlier detection
outlier_detector = IsolationForest(contamination=0.05)  # Adjust the contamination parameter as needed
outliers = outlier_detector.fit_predict(X_tfidf)

# Split the data into training and testing sets
X = data2['cleaned_tokens'].apply(lambda x: ' '.join(x))  # Features
y = data2['Accepted Offer']  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create a Naive Bayes model for text classification
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_tfidf, y_train)

# Define a Z-score threshold for outlier removal
z_score_threshold = 1.6

# Calculate the absolute Z-scores for each feature
z_scores = stats.zscore(features)

# Create a mask to filter out outliers
outlier_mask = (abs(z_scores) < z_score_threshold).all(axis=1)

# Apply the mask to the features and labels
features = features[outlier_mask]
labels = labels[outlier_mask]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier for other features
classifier = GaussianNB()
classifier.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model')
def model():
    if not session.get('logged_in'):
       return render_template('login.html')
    else:
     return render_template('model.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data from HTML
    ask = float(request.form['askAmount'])
    equity = float(request.form['equity'])
    valuationRequested = float(request.form['valuationRequested'])
    train_test_text = request.form['businessDescription']

    user_input_tokens = word_tokenize(train_test_text)
    user_input_lemmatized = lemmatize_sentence(user_input_tokens)
    user_input_cleaned = remove_noise(user_input_tokens, stopwords.words('english'))

    # Vectorize the user input using the same TF-IDF vectorizer
    user_input_tfidf = tfidf_vectorizer.transform([' '.join(user_input_cleaned)])

    # Make predictions with the Naive Bayes model for text classification
    user_input_text_prediction = naive_bayes_model.predict(user_input_tfidf)

    # Use your trained classifier to predict the output based on other features
    input_data = {'Ask(in Lakhs)': [ask], 'Equity': [equity], 'Valuation Requested': [valuationRequested], 'Text Classification Prediction': [user_input_text_prediction[0]]}
    output = classifier.predict(pd.DataFrame(input_data))

    if output[0] == 1:
        prediction = f"Hello Investor, After analyzing your data, it is suggested to invest in this idea."
    else:
        prediction = f"Hello Investor, After analyzing your data, it is not suggested to invest in this idea."
    return render_template('result.html', prediction=prediction)
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Kunal0209@'
app.config['MYSQL_DB'] = 'kunalkp'
mysql = MySQL(app)

app.secret_key = '02092002'

@app.route('/register', methods=['POST'])
def register():
    try:
        if request.method == 'POST':
            username = request.form['username']
            email = request.form['email']
            password1 = request.form['password1']

            # Create a cursor
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM users12 WHERE username = %s", (username,))
            existing_user = cur.fetchone()

            if existing_user:
                # Username already exists, generate a message
                message = 'Username already exists!!'
                return render_template('signup.html', message=message)

            # Insert the user data into the database (use 'users' as the table name)
            cur.execute("INSERT INTO users12 (username, email, password1) VALUES (%s, %s, %s)", (username, email, password1))

            # Commit changes
            mysql.connection.commit()

            # Close the cursor
            cur.close()

            return render_template('login.html')  # Ensure the correct template name
    except Exception as e:
        print(e)  # or log the error
        return "An error occurred", 500  # Return a 500 status code
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST', 'GET'])
def login_post():
    if request.method == 'POST':
        username = request.form['username']
        password1 = request.form['password1']

        try:
            # Check the username and password in your database
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT username, password1 FROM users12 WHERE username = %s", (username,))
            user = cursor.fetchone()

            if user and user[1] == password1:
                session['logged_in'] = True
                return render_template('model.html')
            else:
                message = 'Invalid User Name or Password!!'
                return render_template('login.html', message=message)
        except Exception as e:
            return f"An error occurred: {str(e)}"
        finally:
            cursor.close()
    return render_template('login.html')

@app.route('/logout')
def logout():
    return render_template('logout.html')

@app.route('/logout1', methods=['GET', 'POST'])
def logout1():
     session.pop('logged_in', None)
     return  render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
TWILIO_ACCOUNT_SID = 'ACee1abda130f06624b024b1828dfd519c'
TWILIO_AUTH_TOKEN = 'c3892d6c1a3247669a02e3bc09939b96'

# Replace these with your Twilio phone numbers
TWILIO_PHONE_NUMBER = '+12294412156'
YOUR_PHONE_NUMBER = '+91 99778 53525'

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
@app.route('/call', methods=['POST'])
def call():
    try:
        call = client.calls.create(
            to=YOUR_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            url='http://demo.twilio.com/docs/voice.xml'  # You can provide your own TwiML URL for the call
        )
    except Exception as e:
        return "Error: " + str(e)

    return "Calling..."

@app.route('/feedback1')
def feedback():
    return render_template('feedback1.html')
to_email = 'kunalkp5526@gmail.com'
@app.route('/send-email', methods=['POST'])
def send_email():
    user_name = request.form['user_name']
    message = request.form['message']

    # Replace these with your email server and credentials
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'kunalkp5526@gmail.com'
    smtp_password = 'swvo ciag ujjl ldoo'

    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = to_email
    msg['Subject'] = 'Message from ' + user_name

    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, to_email, msg.as_string())
        server.quit()
        return 'Email sent successfully!'
    except Exception as e:
        return 'Email could not be sent: ' + str(e)


if __name__ == '__main__':
    app.run(debug=False)
    
