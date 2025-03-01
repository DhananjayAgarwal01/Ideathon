from flask import Flask, render_template, request, flash, redirect, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from joblib import load
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agriprice.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
db = SQLAlchemy(app)

# ********************************** Database Models **********************************
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)

# ********************************** Main Routes **********************************
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/contact', methods=['POST'])
def contact():
    try:
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        description = request.form['description']

        new_message = ContactMessage(
            name=name,
            email=email,
            phone=phone,
            description=description
        )

        db.session.add(new_message)
        db.session.commit()

        flash('Thank you for your message! We will get back to you soon.', 'success')
        return redirect(url_for('home'))
    except Exception as e:
        flash('An error occurred while sending your message. Please try again.', 'error')
        return redirect(url_for('home'))

# ********************************** Prediction Functions **********************************
def load_data():
    try:
        df = pd.read_csv('static/models/ideathon_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_models():
    models = {}
    try:
        for commodity in ['onion', 'potato', 'tomato']:
            model_path = f'static/models/{commodity}_model.joblib'
            models[commodity] = load(model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
    return models

# Initialize models
commodity_models = load_models()

# Base prices for wholesale
pulse_base_prices = {
    'gram': 60,
    'tur': 85,
    'urad': 90,
    'moong': 95,
    'masur': 80
}

# ********************************** Prediction Routes **********************************
@app.route('/predict_vegetables')
def predict_vegetables_form():
    return render_template('predict_vegetables.html')

@app.route('/predict_wholesale')
def predict_wholesale_form():
    return render_template('predict_wholesale.html')

@app.route('/predict_vegetables', methods=['POST'])
def predict_vegetables():
    try:
        commodity = request.form['commodity']
        date_str = request.form['date']
        market = request.form['market']
        quantity = float(request.form['quantity'])
        
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        df = load_data()
        if df is None:
            return jsonify({'error': 'Unable to load historical data'}), 500
            
        features = pd.DataFrame({
    'year': [date.year],
    'month': [date.month],
    'week': [date.isocalendar()[1]],
    f'lag_1_{commodity}': [df[commodity].iloc[-1]],
    f'rolling_mean_7_{commodity}': [df[commodity].rolling(window=7).mean().iloc[-1]],
    # Add missing features here (check model.feature_names_in_)
})

        
        try:
            model = load('static/models/final_retailprice_prediction_model.joblib')
            predicted_price = model.predict(features)[0]
            confidence = min(95, int(model.feature_importances_.mean() * 100))
            
            historical_dates = df.tail(30)['date'].dt.strftime('%Y-%m-%d').tolist()
            historical_prices = df[commodity].tail(30).tolist()
            
            future_dates = [(date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            future_features = pd.DataFrame({
                'year': [(date + timedelta(days=i)).year for i in range(7)],
                'month': [(date + timedelta(days=i)).month for i in range(7)],
                'week': [(date + timedelta(days=i)).isocalendar()[1] for i in range(7)],
                f'lag_1_{commodity}': [df[commodity].iloc[-1]] * 7,
                f'rolling_mean_7_{commodity}': [df[commodity].rolling(window=7).mean().iloc[-1]] * 7
            })
            future_prices = model.predict(future_features).tolist()
            
            avg_price = df[commodity].mean()
            price_trend = 'Increasing' if predicted_price > avg_price else 'Decreasing'
            volatility = 'High' if abs(predicted_price - avg_price) > avg_price * 0.2 else 'Moderate'
            
            market_insights = f"The {commodity} prices are showing a {price_trend.lower()} trend. "
            market_insights += f"Price volatility is {volatility.lower()}. "
            
            total_price = predicted_price * quantity
            
            additional_insights = [
                f"Historical average price: ₹{avg_price:.2f}/kg",
                f"Current market trend: {price_trend}",
                f"Price volatility: {volatility}",
                f"Total price for {quantity}kg: ₹{total_price:.2f}",
                f"Recommended action: {'Stock up' if price_trend == 'Increasing' else 'Regular purchase'}"
            ]
            
            return render_template('prediction_result.html',
                                commodity=commodity,
                                date=date_str,
                                market=market,
                                quantity=quantity,
                                unit='kg',
                                predicted_price=predicted_price,
                                total_price=total_price,
                                confidence=confidence,
                                dates=historical_dates + future_dates,
                                historical_prices=historical_prices + future_prices,
                                predicted_prices=[None] * len(historical_dates) + future_prices,
                                market_insights=market_insights,
                                additional_insights=additional_insights)
                                
        except Exception as e:
            return jsonify({'error': f'Error loading model: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_wholesale', methods=['POST'])
def predict_wholesale():
    try:
        commodity = request.form['commodity']
        date_str = request.form['date']
        market = request.form['market']
        quantity = float(request.form['quantity'])
        
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        df = pd.read_csv('static/models/ideathon_wholesale_dataset.csv')  # Load wholesale dataset
        df['date'] = pd.to_datetime(df['date'])
        
        if df is None:
            return jsonify({'error': 'Unable to load historical data'}), 500
        
        features = pd.DataFrame({
            'year': [date.year],
            'month': [date.month],
            'week': [date.isocalendar()[1]],
            f'lag_1_{commodity}': [df[commodity].iloc[-1]],
            f'rolling_mean_7_{commodity}': [df[commodity].rolling(window=7).mean().iloc[-1]]
        })
        
        try:
            model = load(f'static/models/final_wholesaleprice_prediction_model.joblib')
            predicted_price = model.predict(features)[0]
            confidence = min(95, int(model.feature_importances_.mean() * 100))
            
            historical_dates = df.tail(30)['date'].dt.strftime('%Y-%m-%d').tolist()
            historical_prices = df[commodity].tail(30).tolist()
            
            future_dates = [(date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            future_features = pd.DataFrame({
                'year': [(date + timedelta(days=i)).year for i in range(7)],
                'month': [(date + timedelta(days=i)).month for i in range(7)],
                'week': [(date + timedelta(days=i)).isocalendar()[1] for i in range(7)],
                f'lag_1_{commodity}': [df[commodity].iloc[-1]] * 7,
                f'rolling_mean_7_{commodity}': [df[commodity].rolling(window=7).mean().iloc[-1]] * 7
            })
            future_prices = model.predict(future_features).tolist()
            
            avg_price = df[commodity].mean()
            price_trend = 'Increasing' if predicted_price > avg_price else 'Decreasing'
            volatility = 'High' if abs(predicted_price - avg_price) > avg_price * 0.2 else 'Moderate'
            
            market_insights = f"The {commodity} prices are showing a {price_trend.lower()} trend. "
            market_insights += f"Price volatility is {volatility.lower()}. "
            
            total_price = predicted_price * quantity
            
            additional_insights = [
                f"Historical average price: ₹{avg_price:.2f}/kg",
                f"Current market trend: {price_trend}",
                f"Price volatility: {volatility}",
                f"Total price for {quantity}kg: ₹{total_price:.2f}",
                f"Recommended action: {'Stock up' if price_trend == 'Increasing' else 'Regular purchase'}"
            ]
            
            return render_template('prediction_result.html',
                                commodity=commodity,
                                date=date_str,
                                market=market,
                                quantity=quantity,
                                unit='kg',
                                predicted_price=predicted_price,
                                total_price=total_price,
                                confidence=confidence,
                                dates=historical_dates + future_dates,
                                historical_prices=historical_prices + future_prices,
                                predicted_prices=[None] * len(historical_dates) + future_prices,
                                market_insights=market_insights,
                                additional_insights=additional_insights)
                                
        except Exception as e:
            return jsonify({'error': f'Error loading model: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)