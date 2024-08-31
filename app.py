from flask import Flask, render_template, request, flash, redirect
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import re

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///SustainaWatt.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'SustainaWatt'
db = SQLAlchemy(app)

class App1(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    airtemperature = db.Column(db.Float(), nullable=False)
    pressure = db.Column(db.Float(), nullable=False)
    windspeed = db.Column(db.Float(), nullable=False)

# ********************************** DB User table **********************************
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstName = db.Column(db.String(15), nullable=False)
    lastName = db.Column(db.String(15), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

# ********************************** DB Contact Us table **********************************
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)

# ********************************** Creates all DB tables **********************************
with app.app_context():
    db.create_all()
    

name_pattern = re.compile(r'^[a-zA-Z]{1,15}$')
username_pattern = re.compile(r'^[a-z0-9_.]{6,}$')
email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@(?:gmail|yahoo|outlook)\.com$')
password_pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')

@app.route('/')

def index():
    return render_template('index.html')

# Sign up page
@app.route('/signup')
def signup():
    return render_template('signup.html')

# Forgot password page
@app.route('/forgotpassword')
def forgotpassword():
    return render_template('forgotpassword.html')

# Home page
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/app1', methods=['GET','POST'])
def app1():
    if request.method == 'POST':
        air_temperature = float(request.form['airtemperature'])
        pressure = float(request.form['pressure'])
        wind_speed = float(request.form['windspeed'])

        new_App1 = App1(airtemperature=air_temperature, pressure=pressure, windspeed=wind_speed)

        db.session.add(new_App1)
        db.session.commit()

        user_response_app1 = {
            'AirTemp': air_temperature,
            'Pressure': pressure,
            'WindSpeed': wind_speed
        }

        # Creating a feature vector for the model input
        label_encoder = LabelEncoder()
        for column in user_response_app1.keys():
            user_response_app1[column] = label_encoder.fit_transform([user_response_app1[column]])[0]

        features = [user_response_app1[column] for column in user_response_app1.keys()]
        user_input = np.array(features).reshape(1, -1)

        # Using the loaded model to predict the screening score
        loaded_model = joblib.load('static/models/final_power_gen_model.joblib', mmap_mode=None)
        user_screening_score = loaded_model.predict_proba(user_input)[:, 1]

        # Outputting the screening score
        print(user_screening_score)
        flash(f"User Screening Score: {user_screening_score}")

    return render_template('app1.html')

@app.route('/app2')
def app2():
    return render_template('app2.html')

@app.route('/signup',  methods=['GET', 'POST']) # Accept both GET and POST requests
def sign_up():
    if request.method == 'POST':
        firstName = request.form['firstname']
        lastName = request.form['lastname']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if the first name and last name meet the criteria
        if not name_pattern.match(firstName) or not name_pattern.match(lastName):
            flash('First name and last name must be between 1 and 15 characters long and contain only letters.', 'error')
            return redirect('/signup')


        # Check if the email meets the criteria
        if not email_pattern.match(email):
            flash('Email must be in a valid format (@gmail.com, @yahoo.com, @outlook.com, etc.)', 'error')
            return redirect('/signup')

        # Check if the password meets the criteria
        if not password_pattern.match(password):
            flash('Password must contain at least 8 characters, 1 capital letter, 1 small letter, and 1 symbol.', 'error')
            return redirect('/signup')

        # Check if the username or email already exists
        if User.query.filter_by(username=username).first() is not None:
            flash('User already exists! Choose a different username.', 'error')
            return redirect('/signup')
        elif User.query.filter_by(email=email).first() is not None:
            flash('Email already exists! Use a different email address.', 'error')
            return redirect('/signup')

        # If all validation passes, create a new user
        new_user = User(firstName=firstName, lastName=lastName, username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Now, user can sign in.', 'success')
        return redirect('/signin')

    # If it's a GET request, just render the sign_up form
    return render_template('signup.html')


@app.route('/signin', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find user by username
        user = User.query.filter_by(username=username).first()

        if user:
            # Check if the password matches
            if user.password == password:
                flash('sign_in successful!', 'success')
                return redirect('/home')
            else:
                flash('Incorrect password. Please try again.', 'error')
                return redirect('/signin')  # Redirect after flashing error message
        else:
            flash('User not found. Please sign up first.', 'error')
            return redirect('/signup')

    return render_template('signin.html')



if __name__ == "__main__":
    app.run(debug=True)