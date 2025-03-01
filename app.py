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
        retail_df = pd.read_csv('static/models/ideathon_dataset.csv')
        wholesale_df = pd.read_csv('static/models/ideathon_wholesale_dataset.csv')
        retail_df['date'] = pd.to_datetime(retail_df['date'])
        wholesale_df['date'] = pd.to_datetime(wholesale_df['date'])
        return retail_df, wholesale_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
# Initialize data
retail_df, wholesale_df = load_data()

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
# ********************************** Prediction Routes **********************************
@app.route('/predict_vegetables')
def predict_vegetables_form():
    return render_template('predict_vegetables.html')

@app.route('/predict_wholesale')
def predict_wholesale_form():
    return render_template('predict_wholesale.html')

# Fix the predict_vegetables function - remove duplicate decorator
@app.route('/predict_vegetables', methods=['POST'])
def predict_vegetables():
    try:
        commodity = request.form['commodity']
        date_str = request.form['date']
        market = request.form['market']
        quantity = float(request.form['quantity'])
        
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Get retail dataset
        retail_df, _ = load_data()
        if retail_df is None:
            return jsonify({'error': 'Unable to load historical data'}), 500
            
        # Ensure we're using the correct model for retail prices
        try:
            # Use the commodity-specific model from the loaded models
            model = commodity_models.get(commodity)
            
            # If commodity-specific model not found, fall back to generic retail model
            if model is None:
                model = load('static/models/final_retailprice_prediction_model.joblib')
                
            # Prepare features - make sure these match what your model expects
            features = pd.DataFrame({
                'year': [date.year],
                'month': [date.month],
                'week': [date.isocalendar()[1]],
                f'lag_1_{commodity}': [retail_df[commodity].iloc[-1]],
                f'rolling_mean_7_{commodity}': [retail_df[commodity].rolling(window=7).mean().iloc[-1]],
                # You may need additional features based on your model
            })
            
            # Make prediction
            predicted_price = model.predict(features)[0]
            confidence = min(95, int(model.feature_importances_.mean() * 100))
            
            # Get historical data
            historical_dates = retail_df.tail(30)['date'].dt.strftime('%Y-%m-%d').tolist()
            historical_prices = retail_df[commodity].tail(30).tolist()
            
            # Predict future prices
            future_dates = [(date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            future_features = pd.DataFrame({
                'year': [(date + timedelta(days=i)).year for i in range(7)],
                'month': [(date + timedelta(days=i)).month for i in range(7)],
                'week': [(date + timedelta(days=i)).isocalendar()[1] for i in range(7)],
                f'lag_1_{commodity}': [retail_df[commodity].iloc[-1]] * 7,
                f'rolling_mean_7_{commodity}': [retail_df[commodity].rolling(window=7).mean().iloc[-1]] * 7
                # Add the same features as above
            })
            future_prices = model.predict(future_features).tolist()
            
            # Calculate insights
            avg_price = retail_df[commodity].mean()
            price_trend = 'Increasing' if predicted_price > avg_price else 'Decreasing'
            volatility = 'High' if abs(predicted_price - avg_price) > avg_price * 0.2 else 'Moderate'
            
            market_insights = f"The retail {commodity} prices are showing a {price_trend.lower()} trend. "
            market_insights += f"Price volatility is {volatility.lower()}. "
            
            total_price = predicted_price * quantity
            
            additional_insights = [
                f"Historical average retail price: ₹{avg_price:.2f}/kg",
                f"Current retail market trend: {price_trend}",
                f"Price volatility: {volatility}",
                f"Total retail price for {quantity}kg: ₹{total_price:.2f}",
                f"Recommended action: {'Stock up' if price_trend == 'Increasing' else 'Regular purchase'}"
            ]
            
            # Debug message
            print(f"RETAIL PREDICTION: {predicted_price} for {commodity}")
            
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
            import traceback
            print(f"Error in retail prediction: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error in retail prediction: {str(e)}'}), 500
            
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
        
        # Get wholesale dataset
        _, wholesale_df = load_data()
        if wholesale_df is None:
            return jsonify({'error': 'Unable to load wholesale data'}), 500
        
        try:
            # Always use the specific wholesale model
            model = load('static/models/final_wholesaleprice_prediction_model.joblib')
            
            # Prepare features for wholesale prediction
            features = pd.DataFrame({
                'year': [date.year],
                'month': [date.month],
                'week': [date.isocalendar()[1]],
                f'lag_1_{commodity}': [wholesale_df[commodity].iloc[-1]],
                f'rolling_mean_7_{commodity}': [wholesale_df[commodity].rolling(window=7).mean().iloc[-1]]
                # Make sure these features match what your wholesale model expects
            })
            
            # Make prediction
            predicted_price = model.predict(features)[0]
            confidence = min(95, int(model.feature_importances_.mean() * 100))
            
            # Get historical data
            historical_dates = wholesale_df.tail(30)['date'].dt.strftime('%Y-%m-%d').tolist()
            historical_prices = wholesale_df[commodity].tail(30).tolist()
            
            # Predict future prices
            future_dates = [(date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            future_features = pd.DataFrame({
                'year': [(date + timedelta(days=i)).year for i in range(7)],
                'month': [(date + timedelta(days=i)).month for i in range(7)],
                'week': [(date + timedelta(days=i)).isocalendar()[1] for i in range(7)],
                f'lag_1_{commodity}': [wholesale_df[commodity].iloc[-1]] * 7,
                f'rolling_mean_7_{commodity}': [wholesale_df[commodity].rolling(window=7).mean().iloc[-1]] * 7
            })
            future_prices = model.predict(future_features).tolist()
            
            # Calculate insights
            avg_price = wholesale_df[commodity].mean()
            price_trend = 'Increasing' if predicted_price > avg_price else 'Decreasing'
            volatility = 'High' if abs(predicted_price - avg_price) > avg_price * 0.2 else 'Moderate'
            
            market_insights = f"The wholesale {commodity} prices are showing a {price_trend.lower()} trend. "
            market_insights += f"Price volatility is {volatility.lower()}. "
            
            total_price = predicted_price * quantity
            
            additional_insights = [
                f"Historical average wholesale price: ₹{avg_price:.2f}/kg",
                f"Current wholesale market trend: {price_trend}",
                f"Price volatility: {volatility}",
                f"Total wholesale price for {quantity}kg: ₹{total_price:.2f}",
                f"Recommended action: {'Stock up' if price_trend == 'Increasing' else 'Regular purchase'}"
            ]
            
            # Debug message
            print(f"WHOLESALE PREDICTION: {predicted_price} for {commodity}")
            
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
            import traceback
            print(f"Error in wholesale prediction: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error in wholesale prediction: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)