import os
import warnings
import numpy as np
import pandas as pd
import pickle
import joblib
import random
import gc
from flask import Flask, render_template, flash, redirect, url_for, session, request, jsonify
from forms import RegisterForm, LoginForm
from models import db, User
from flask_bcrypt import Bcrypt
from functools import wraps
import pandas as pd
import numpy as np
import joblib
import os
import pickle
import random
import warnings

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'ASLDFJASDLFADS')

# Database configuration - use environment variable for production
database_url = os.environ.get('DATABASE_URL')
if database_url:
    # Production database (Render will provide this)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    # Use SQLite for deployment/development when PostgreSQL is not available
    # Create a data directory if it doesn't exist
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Use absolute path for SQLite database
    db_path = os.path.join(data_dir, 'fitintel.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    print(f"💾 Database path: {db_path}")

# Additional database configuration for better persistence
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_timeout': 20,
    'pool_recycle': -1,
    'pool_pre_ping': True
}

# Initialize extensions
bcrypt = Bcrypt(app)
db.init_app(app)

# Suppress warnings
warnings.filterwarnings('ignore')

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# ==================== Authentication Decorators ====================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== Model Loading Functions ====================
# Global model caches for lazy loading
_exercise_models_cache = None
_exercise_models1_cache = None
_unique_exercises_cache = None
_food_models_cache = {}

def get_exercise_models():
    """Exercise models disabled - only fever diet planner is active"""
    print("⚠️ Exercise recommendations are disabled. Only fever diet planner is active.")
    return {}, {}, []

def get_food_models(model_type=None):
    """Lazy load food recommendation models by type - FEVER ONLY"""
    global _food_models_cache
    
    # Only allow fever model type for this simplified version
    if model_type != 'fever':
        print(f"⚠️ Only fever diet planner is active. Requested: {model_type}")
        return {}
    
    if model_type and model_type in _food_models_cache:
        return _food_models_cache[model_type]
    
    print(f"Loading {model_type} food models on-demand...")
    
    if model_type == 'fever' and 'fever' not in _food_models_cache:
        _food_models_cache['fever'] = {}
        try:
            _food_models_cache['fever']['breakfast_model'] = joblib.load('models/breakfast_model_fever.pkl')
            _food_models_cache['fever']['lunch_model'] = joblib.load('models/lunch_model_fever.pkl')
            _food_models_cache['fever']['dinner_model'] = joblib.load('models/dinner_model_fever.pkl')
            _food_models_cache['fever']['meal_encoders'] = joblib.load('models/meal_encoders_fever.pkl')
            _food_models_cache['fever']['gender_encoder'] = joblib.load('models/gender_encoder_fever.pkl')
            print("✅ Loaded fever food models")
        except Exception as e:
            print(f"❌ Warning: Could not load fever food models: {e}")
    
    # Clean up memory after model loading
    gc.collect()
    
    return _food_models_cache.get(model_type, {})

# Load all models at startup
with app.app_context():
    # Try to create database tables with better error handling
    try:
        # First, ensure we can connect to the database
        db.engine.execute("SELECT 1")
        print("✅ Database connection successful")
        
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully")
        
        # Verify tables exist by checking User table
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"📊 Found {len(tables)} database tables: {tables}")
        
        # Check if we have any existing users
        user_count = User.query.count()
        print(f"👥 Current user count: {user_count}")
        
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        print("🔧 This might be due to database file permissions or connection issues")
    
    print("🚀 FitIntel app starting - FEVER DIET PLANNER ONLY")
    print("🌡️ Only fever diet recommendations are active")
    print("⚠️ Exercise, heart, diabetes, and general diet features are temporarily disabled")

# ==================== Helper Functions ====================
def get_bmi_category(bmi):
    """Return BMI category based on BMI value"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def map_categories_to_food_items(categories, meal_type, dietary_preference):
    """Map food categories back to actual food items (for diabetes)"""
    defaults = {
        'breakfast': {
            'Vegetarian': ['Vegetable Oats', 'Sprouts Salad', 'Ragi Dosa'],
            'Non-Vegetarian': ['Egg Whites with Vegetables', 'Chicken Sandwich with Whole Grain Bread', 'Fish with Vegetables']
        },
        'lunch': {
            'Vegetarian': ['Brown Rice with Dal', 'Multigrain Roti with Vegetable Curry', 'Ragi Roti with Sambar'],
            'Non-Vegetarian': ['Grilled Chicken with Brown Rice', 'Fish Curry with Brown Rice', 'Egg Curry with Multigrain Roti']
        },
        'dinner': {
            'Vegetarian': ['Vegetable Soup', 'Vegetable Khichdi', 'Vegetable Salad'],
            'Non-Vegetarian': ['Chicken Soup', 'Grilled Fish with Vegetables', 'Egg White Omelet with Vegetables']
        }
    }
    
    try:
        df = pd.read_csv('diabetes_food_recommendations.csv')
        
        if dietary_preference == 'Vegetarian':
            df = df[df['dietary_preference'] == 'Vegetarian']
        else:
            df = df[df['dietary_preference'] == 'Non-Vegetarian']
        
        if meal_type == 'breakfast':
            meal_items_col = 'breakfast_items'
        elif meal_type == 'lunch':
            meal_items_col = 'lunch_items'
        else:
            meal_items_col = 'dinner_items'
        
        matching_rows = []
        if categories and categories.strip():
            categories_list = categories.split()
            
            for _, row in df.iterrows():
                items = str(row[meal_items_col])
                if all(category in items.lower() for category in categories_list):
                    matching_rows.append(row[meal_items_col])
            
            if not matching_rows:
                for _, row in df.iterrows():
                    items = str(row[meal_items_col])
                    if any(category in items.lower() for category in categories_list):
                        matching_rows.append(row[meal_items_col])
        
        if matching_rows:
            all_items = []
            for items_str in matching_rows:
                items = items_str.split(', ')
                all_items.extend(items)
            return list(set(all_items))[:3]
    except:
        pass
    
    return defaults[meal_type][dietary_preference]

def predict_food_recommendations(age, weight, gender, fasting_blood_sugar, hba1c, diabetes_type, dietary_preference):
    """Diabetes food recommendations disabled - only fever diet planner is active"""
    return {
        'breakfast': ['Fever diet planner only'],
        'lunch': ['Please use fever recommendations'],
        'dinner': ['Diabetes feature coming soon']
    }

# ==================== Authentication Routes ====================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Database and system health check endpoint"""
    try:
        # Check database connection
        db.engine.execute("SELECT 1")
        
        # Count users
        user_count = User.query.count()
        
        # Get database file info (if SQLite)
        db_info = {}
        if 'sqlite' in app.config.get('SQLALCHEMY_DATABASE_URI', ''):
            db_uri = app.config['SQLALCHEMY_DATABASE_URI']
            if 'sqlite:///' in db_uri:
                db_path = db_uri.replace('sqlite:///', '')
                if os.path.exists(db_path):
                    stat = os.stat(db_path)
                    db_info = {
                        'db_path': db_path,
                        'db_exists': True,
                        'db_size_bytes': stat.st_size,
                        'db_modified': stat.st_mtime
                    }
                else:
                    db_info = {'db_path': db_path, 'db_exists': False}
        
        return jsonify({
            'status': 'healthy',
            'mode': 'fever-diet-only',
            'active_features': ['fever_diet_planner'],
            'disabled_features': ['exercise_recommendations', 'heart_diet', 'diabetes_diet', 'general_diet'],
            'database': 'connected',
            'user_count': user_count,
            'database_info': db_info,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'database': 'disconnected',
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat()
        }), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Please check email and password', 'danger')
    return render_template("login.html", form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('home'))
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=hashed_password,
        )
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created!', 'success')
        return redirect(url_for('login'))
    return render_template("register.html", form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    session.pop('user_id')
    session.pop('username')
    flash('You have been logged out!', 'success')
    return redirect(url_for("home"))

# ==================== Content Pages ====================
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/weight')
@login_required
def weight_index():
    return render_template('weight.html')

@app.route('/food_recommendation')
@login_required
def food_recommendation():
    return render_template('food_recommendation.html')

@app.route('/exercise_recommendation')
@login_required
def exercise_recommendation():
    return render_template('exercise_recommendation.html')

@app.route('/fitness')
@login_required
def fitness_index():
    return render_template('fitness.html')

# ==================== Exercise Recommendation Routes ====================
@app.route('/predict1', methods=['POST'])
@login_required
def predict1():
    """Exercise recommendations disabled - only fever diet planner is active"""
    return jsonify({
        'success': False,
        'error': 'Exercise recommendations are temporarily disabled. Only fever diet planner is currently active.',
        'message': 'Please use the Fever Diet Planner for food recommendations during fever.'
    })

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Exercise recommendations disabled - only fever diet planner is active"""
    return jsonify({
        'success': False,
        'error': 'Exercise recommendations are temporarily disabled. Only fever diet planner is currently active.',
        'message': 'Please use the Fever Diet Planner for food recommendations during fever.'
    })

# ==================== Food Recommendation Routes ====================
@app.route('/fever')
@login_required
def fever_home():
    return render_template('fever.html')

@app.route('/predict_fever', methods=['POST'])
@login_required
def predict_fever():
    fever_models = get_food_models('fever')
    if not fever_models or not all(key in fever_models for key in ['breakfast_model', 'lunch_model', 'dinner_model']):
        return jsonify({'error': 'Fever models not loaded'}), 500
    
    data = request.form
    age = float(data['age'])
    weight = float(data['weight'])
    gender = data['gender']
    fever_level = float(data['fever_level'])

    input_data = pd.DataFrame({
        'age': [age],
        'weight': [weight],
        'gender': [gender],
        'fever_level': [fever_level]
    })

    input_data['gender'] = fever_models['gender_encoder'].transform(input_data['gender'])

    breakfast_pred = fever_models['breakfast_model'].predict(input_data)[0]
    lunch_pred = fever_models['lunch_model'].predict(input_data)[0]
    dinner_pred = fever_models['dinner_model'].predict(input_data)[0]

    breakfast_rec = fever_models['meal_encoders']['breakfast'].inverse_transform([breakfast_pred])[0]
    lunch_rec = fever_models['meal_encoders']['lunch'].inverse_transform([lunch_pred])[0]
    dinner_rec = fever_models['meal_encoders']['dinner'].inverse_transform([dinner_pred])[0]

    return jsonify({
        'breakfast': breakfast_rec,
        'lunch': lunch_rec,
        'dinner': dinner_rec
    })

@app.route('/heart')
@login_required
def heart_home():
    return render_template('heart.html')

@app.route('/predict_heart', methods=['POST'])
@login_required
def predict_heart():
    """Heart recommendations disabled - only fever diet planner is active"""
    result = {
        'risk_category': 'Feature Disabled',
        'breakfast_recommendation': 'Only fever diet planner is currently active',
        'lunch_recommendation': 'Please use the Fever Diet Planner',
        'dinner_recommendation': 'Heart recommendations coming soon'
    }
    
    return render_template('heart.html', prediction=result)

@app.route('/diabetes')
@login_required
def diabetes_home():
    return render_template('diabetes.html')

@app.route('/predict_diabetes', methods=['POST'])
@login_required
def predict_diabetes():
    """Diabetes recommendations disabled - only fever diet planner is active"""
    return jsonify({
        'status': 'disabled',
        'message': 'Diabetes recommendations are temporarily disabled. Only fever diet planner is currently active.',
        'recommendations': {
            'breakfast': ['Please use the Fever Diet Planner'],
            'lunch': ['Diabetes recommendations coming soon'],
            'dinner': ['Only fever diet planning is active']
        }
    })

@app.route('/diet')
@login_required
def diet_home():
    return render_template('diet.html')

@app.route('/recommend', methods=['POST'])
@login_required
def recommend_diet():
    """General diet recommendations disabled - only fever diet planner is active"""
    return jsonify({
        'breakfast': 'Only fever diet planner is currently active',
        'lunch': 'Please use the Fever Diet Planner for recommendations',
        'dinner': 'General diet recommendations coming soon',
        'note': 'This feature is temporarily disabled. Use the Fever Diet Planner for food recommendations.'
    })

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
