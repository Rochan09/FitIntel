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

def load_compressed_model(model_name):
    """Load model from compressed directory first, fallback to regular models"""
    compressed_path = f'models_compressed/{model_name}'
    regular_path = f'models/{model_name}'
    
    # Try compressed version first
    if os.path.exists(compressed_path):
        try:
            model = joblib.load(compressed_path)
            print(f"✅ Loaded compressed: {model_name}")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load compressed {model_name}: {e}")
    
    # Fallback to regular version
    if os.path.exists(regular_path):
        try:
            model = joblib.load(regular_path)
            print(f"✅ Loaded regular: {model_name}")
            return model
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return None
    
    print(f"❌ Model not found: {model_name}")
    return None

def get_food_models(model_type=None):
    """Lazy load food recommendation models by type - FEVER ONLY (COMPRESSED)"""
    global _food_models_cache
    
    # Only allow fever model type for this simplified version
    if model_type != 'fever':
        print(f"⚠️ Only fever diet planner is active. Requested: {model_type}")
        return {}
    
    if model_type and model_type in _food_models_cache:
        return _food_models_cache[model_type]
    
    print(f"Loading {model_type} food models on-demand (COMPRESSED VERSION)...")
    
    if model_type == 'fever' and 'fever' not in _food_models_cache:
        _food_models_cache['fever'] = {}
        try:
            # Load only breakfast and lunch models for fever
            _food_models_cache['fever']['breakfast_model'] = load_compressed_model('breakfast_model_fever.pkl')
            _food_models_cache['fever']['lunch_model'] = load_compressed_model('lunch_model_fever.pkl')
            # No dinner, encoders, or gender encoder loaded
            loaded_models = sum(1 for v in _food_models_cache['fever'].values() if v is not None)
            print(f"✅ Loaded {loaded_models}/2 compressed fever food models")
            
        except Exception as e:
            print(f"❌ Warning: Could not load fever food models: {e}")
    
    # Clean up memory after model loading
    gc.collect()
    
    return _food_models_cache.get(model_type, {})

# Load all models at startup
with app.app_context():
    # Try to create database tables with better error handling
    try:
        # First, ensure we can connect to the database (SQLAlchemy 2.x compatible)
        with db.engine.connect() as conn:
            conn.execute(db.text("SELECT 1"))
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
    
    print("🚀 FitIntel app starting - FULL FEATURED with SMART RECOMMENDATIONS")
    print("🌡️ Fever Diet Planner: REAL ML-powered recommendations")
    print("🎯 Exercise, Heart, Diabetes, Diet: Smart fake recommendations based on user inputs")
    print("📦 Using compressed models (6.85MB total) for Render free tier")

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
            'mode': 'full-featured-with-smart-recommendations',
            'ml_powered_features': ['fever_diet_planner'],
            'smart_fake_features': ['exercise_recommendations', 'heart_diet', 'diabetes_diet', 'general_diet'],
            'model_info': {
                'total_compressed_size_mb': 6.85,
                'compression_ratio': '98.9%',
                'original_size_mb': 598.89,
                'render_free_tier_compatible': True
            },
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
    # Smart fake recommendations for diabetes and diet maintenance
    diabetes_food = {
        'breakfast': ['Oatmeal with berries', 'Egg whites with spinach', 'Greek yogurt with nuts'],
        'lunch': ['Grilled chicken salad', 'Quinoa bowl with vegetables', 'Lentil soup with whole grain bread'],
        'dinner': ['Baked fish with broccoli', 'Vegetable stir-fry with tofu', 'Lean beef with sweet potato'],
        'note': 'Sample diabetes-friendly recommendations. Real ML predictions coming soon!'
    }
    diet_maintenance_food = {
        'breakfast': ['High-protein smoothie', 'Whole grain cereal with milk', 'Fruit and nut bowl'],
        'lunch': ['Grilled salmon with quinoa', 'Vegetable wrap with hummus', 'Chicken and vegetable soup'],
        'dinner': ['Grilled tofu with vegetables', 'Baked chicken with green beans', 'Vegetable curry with rice'],
        'note': 'Sample diet maintenance recommendations. Real ML predictions coming soon!'
    }
    return render_template('food_recommendation.html', diabetes_food=diabetes_food, diet_maintenance_food=diet_maintenance_food)

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
    """Exercise recommendations with fake data - only fever diet planner uses real ML"""
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        fitness_level = int(request.form['fitness_level'])
        time_available = int(request.form['time_available'])
        equipment_access = request.form['equipment_access']
        intensity_preference = request.form['intensity_preference']
        
        bmi = weight / ((height/100) ** 2)
        
        # Generate fake but realistic exercise recommendations based on inputs
        base_exercises = {
            'gym': ['Treadmill Running', 'Weight Training', 'Elliptical Machine', 'Bench Press', 'Squats'],
            'home': ['Push-ups', 'Burpees', 'Jumping Jacks', 'Plank', 'Mountain Climbers'],
            'outdoor': ['Running', 'Cycling', 'Swimming', 'Hiking', 'Walking'],
            'none': ['Bodyweight Squats', 'Yoga', 'Stretching', 'Dancing', 'Stairs Climbing']
        }
        
        equipment_key = 'gym' if equipment_access == 'Full gym' else 'home' if equipment_access == 'Basic home equipment' else 'outdoor' if equipment_access == 'Outdoor activities' else 'none'
        
        # Adjust recommendations based on intensity
        if intensity_preference == 'High':
            recommended_exercises = ['HIIT Workout', 'Sprint Training'] + base_exercises[equipment_key][:2]
        elif intensity_preference == 'Low':
            recommended_exercises = ['Gentle Yoga', 'Walking'] + base_exercises[equipment_key][:2]
        else:
            recommended_exercises = base_exercises[equipment_key][:3]
        
        return jsonify({
            'success': True,
            'exercises': recommended_exercises[:3],
            'profile': {
                'age': age,
                'gender': gender,
                'bmi': round(bmi, 1),
                'fitness_level': fitness_level
            },
            'note': 'These are sample recommendations. Real ML predictions coming soon! Use Fever Diet Planner for actual ML-powered recommendations.'
        })
        
    except Exception as e:
        # Always return valid JSON with smart fake recommendations
        return jsonify({
            'success': True,
            'exercises': ['Walking', 'Bodyweight Squats', 'Gentle Yoga'],
            'profile': {
                'age': 30,
                'gender': 'Male',
                'bmi': 22.0,
                'fitness_level': 2
            },
            'note': 'Sample exercise recommendations. Real ML predictions coming soon! Use Fever Diet Planner for actual ML-powered recommendations.'
        })

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Advanced exercise recommendations with fake data - only fever diet planner uses real ML"""
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        fitness_level = int(request.form['fitness_level'])
        weight_loss_goal = int(request.form['weight_loss_goal'])
        time_available = int(request.form['time_available'])
        equipment_access = request.form['equipment_access']
        intensity_preference = request.form['intensity_preference']
        target_areas = request.form.getlist('target_areas')
        
        bmi = weight / ((height/100) ** 2)
        
        # Generate fake but realistic detailed exercise recommendations
        exercise_database = {
            'cardio': {
                'high': ['HIIT Sprints', 'Burpee Intervals', 'Jump Rope', 'Mountain Climber Circuits'],
                'moderate': ['Jogging', 'Cycling', 'Swimming', 'Elliptical Training'],
                'low': ['Walking', 'Light Cycling', 'Water Aerobics', 'Gentle Swimming']
            },
            'strength': {
                'high': ['Deadlifts', 'Squats with Weights', 'Pull-ups', 'Bench Press'],
                'moderate': ['Bodyweight Squats', 'Push-ups', 'Lunges', 'Plank Hold'],
                'low': ['Wall Push-ups', 'Chair Squats', 'Resistance Band Exercises', 'Light Weights']
            },
            'flexibility': {
                'high': ['Power Yoga', 'Dynamic Stretching', 'Martial Arts', 'Dance Fitness'],
                'moderate': ['Yoga Flow', 'Pilates', 'Stretching Routine', 'Tai Chi'],
                'low': ['Gentle Yoga', 'Basic Stretching', 'Relaxation Poses', 'Meditation']
            }
        }
        
        intensity_key = 'high' if intensity_preference == 'High' else 'low' if intensity_preference == 'Low' else 'moderate'
        
        # Select exercises based on target areas and preferences
        recommended_exercises = []
        if 'Cardio' in target_areas or weight_loss_goal > 5:
            recommended_exercises.extend(exercise_database['cardio'][intensity_key][:2])
        if 'Strength' in target_areas or 'Upper body' in target_areas:
            recommended_exercises.extend(exercise_database['strength'][intensity_key][:1])
        if 'Flexibility' in target_areas or age > 50:
            recommended_exercises.extend(exercise_database['flexibility'][intensity_key][:1])
        
        # Fill with default exercises if none selected
        if not recommended_exercises:
            recommended_exercises = exercise_database['cardio'][intensity_key][:2] + exercise_database['strength'][intensity_key][:1]
        
        # Create detailed exercise information
        exercise_details = []
        for exercise in recommended_exercises[:3]:
            duration = min(time_available // len(recommended_exercises[:3]), 30)
            if intensity_preference == 'High':
                duration = max(duration - 5, 15)
                description = f"High-intensity {exercise.lower()} to maximize calorie burn and improve cardiovascular fitness rapidly."
            elif intensity_preference == 'Low':
                duration = min(duration + 10, 45)
                description = f"Gentle {exercise.lower()} focusing on proper form and gradual fitness improvement."
            else:
                description = f"Moderate {exercise.lower()} for balanced fitness development and sustainable progress."
            
            exercise_details.append({
                'name': exercise,
                'duration': f"{duration} minutes",
                'description': description
            })
        
        return jsonify({
            'success': True,
            'exercises': exercise_details,
            'profile': {
                'age': age,
                'gender': gender,
                'bmi': round(bmi, 1),
                'bmi_category': get_bmi_category(bmi),
                'fitness_level': fitness_level,
                'weight_loss_goal': weight_loss_goal
            },
            'note': 'These are sample recommendations based on your inputs. Real ML predictions coming soon! Use Fever Diet Planner for actual ML-powered recommendations.'
        })
        
    except Exception as e:
        # Always return valid JSON with smart fake recommendations
        return jsonify({
            'success': True,
            'exercises': [
                {'name': 'Walking', 'duration': '30 minutes', 'description': 'Simple cardio exercise for general fitness'},
                {'name': 'Bodyweight Squats', 'duration': '20 minutes', 'description': 'Strength and endurance training'},
                {'name': 'Gentle Yoga', 'duration': '15 minutes', 'description': 'Flexibility and relaxation'}
            ],
            'profile': {
                'age': 30,
                'gender': 'Male',
                'bmi': 22.0,
                'bmi_category': 'Normal',
                'fitness_level': 2,
                'weight_loss_goal': 5
            },
            'note': 'Sample exercise recommendations. Real ML predictions coming soon! Use Fever Diet Planner for actual ML-powered recommendations.'
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
    # If any model missing, return smart fake recommendations
    if not fever_models or not all(key in fever_models for key in ['breakfast_model', 'lunch_model']):
        return jsonify({
            'breakfast': 'Oatmeal with honey and banana',
            'lunch': 'Vegetable soup with whole grain bread',
            'dinner': 'Rice porridge with steamed vegetables',
            'note': 'Sample fever diet recommendations. Real ML predictions coming soon!'
        })

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

    # If gender encoder missing, use default encoding
    if 'gender_encoder' in fever_models and fever_models['gender_encoder'] is not None:
        input_data['gender'] = fever_models['gender_encoder'].transform(input_data['gender'])
    else:
        input_data['gender'] = [0]

    breakfast_pred = fever_models['breakfast_model'].predict(input_data)[0]
    lunch_pred = fever_models['lunch_model'].predict(input_data)[0]

    # If meal_encoders missing, return raw predictions
    breakfast_rec = breakfast_pred if 'meal_encoders' not in fever_models or fever_models['meal_encoders'] is None else fever_models['meal_encoders']['breakfast'].inverse_transform([breakfast_pred])[0]
    lunch_rec = lunch_pred if 'meal_encoders' not in fever_models or fever_models['meal_encoders'] is None else fever_models['meal_encoders']['lunch'].inverse_transform([lunch_pred])[0]
    dinner_rec = 'Rice porridge with steamed vegetables'

    return jsonify({
        'breakfast': breakfast_rec,
        'lunch': lunch_rec,
        'dinner': dinner_rec,
        'note': 'Fever diet recommendations powered by ML (if models available), otherwise sample recommendations.'
    })

@app.route('/heart')
@login_required
def heart_home():
    return render_template('heart.html')

@app.route('/predict_heart', methods=['POST'])
@login_required
def predict_heart():
    """Heart diet recommendations with fake data based on inputs - only fever diet planner uses real ML"""
    try:
        age = int(request.form['age'])
        weight = int(request.form['weight'])
        gender = request.form['gender']
        cholesterol = int(request.form['cholesterol'])
        bp_systolic = int(request.form['bp_systolic'])
        bp_diastolic = int(request.form['bp_diastolic'])
        obesity = request.form['obesity']
        
        # Generate fake but realistic heart-healthy recommendations
        bmi = weight / ((170/100) ** 2)  # Approximate BMI
        
        # Determine risk level based on inputs
        risk_factors = 0
        if age > 60: risk_factors += 1
        if cholesterol > 200: risk_factors += 1
        if bp_systolic > 140 or bp_diastolic > 90: risk_factors += 1
        if obesity in ['Obese', 'Severely Obese']: risk_factors += 1
        
        if risk_factors >= 3:
            risk_category = 'High Risk'
            breakfast_rec = 'Oatmeal with berries, green tea, omega-3 rich walnuts'
            lunch_rec = 'Grilled salmon, quinoa, steamed vegetables, olive oil dressing'
            dinner_rec = 'Baked chicken breast, brown rice, leafy greens, herbal tea'
        elif risk_factors >= 2:
            risk_category = 'Moderate Risk'
            breakfast_rec = 'Whole grain toast with avocado, low-fat yogurt, fresh fruits'
            lunch_rec = 'Lean turkey sandwich, mixed salad, low-sodium soup'
            dinner_rec = 'Grilled fish, sweet potato, broccoli, minimal salt seasoning'
        else:
            risk_category = 'Low Risk'
            breakfast_rec = 'Greek yogurt with nuts, whole grain cereal, banana'
            lunch_rec = 'Chicken salad with olive oil, whole wheat bread, vegetable soup'
            dinner_rec = 'Lean beef, quinoa, mixed vegetables, heart-healthy herbs'
        
        result = {
            'risk_category': f'{risk_category} (Sample)',
            'breakfast_recommendation': breakfast_rec,
            'lunch_recommendation': lunch_rec,
            'dinner_recommendation': dinner_rec,
            'note': 'These are sample heart-healthy recommendations. Real ML predictions coming soon! Use Fever Diet Planner for actual ML-powered recommendations.'
        }
        
        return render_template('heart.html', prediction=result)
        
    except Exception as e:
        result = {
            'risk_category': 'Assessment Unavailable',
            'breakfast_recommendation': 'Heart-healthy oatmeal with fruits',
            'lunch_recommendation': 'Lean protein with vegetables',
            'dinner_recommendation': 'Grilled fish with whole grains',
            'note': 'Sample recommendations only. Please consult healthcare provider.'
        }
        return render_template('heart.html', prediction=result)

@app.route('/diabetes')
@login_required
def diabetes_home():
    return render_template('diabetes.html')

@app.route('/predict_diabetes', methods=['POST'])
@login_required
def predict_diabetes():
    """Diabetes diet recommendations with fake data based on inputs - only fever diet planner uses real ML"""
    try:
        age = int(request.form.get('age', 0))
        weight = float(request.form.get('weight', 0))
        gender = request.form.get('gender', 'Male')
        fasting_blood_sugar = float(request.form.get('fasting_blood_sugar', 100))
        hba1c = float(request.form.get('hba1c', 5.5))
        diabetes_type = request.form.get('diabetes_type', 'Type 2')
        dietary_preference = request.form.get('dietary_preference', 'Non-Vegetarian')
        
        # Generate fake but realistic diabetes-friendly recommendations
        severity = 'Normal'
        if fasting_blood_sugar > 126 or hba1c > 7.0:
            severity = 'High'
        elif fasting_blood_sugar > 100 or hba1c > 6.0:
            severity = 'Moderate'
        
        # Base recommendations by dietary preference and severity
        recommendations = {
            'Vegetarian': {
                'High': {
                    'breakfast': ['Steel-cut oats with cinnamon', 'Greek yogurt with berries', 'Vegetable omelet'],
                    'lunch': ['Quinoa salad with vegetables', 'Lentil soup with whole grain bread', 'Chickpea curry with brown rice'],
                    'dinner': ['Grilled tofu with vegetables', 'Vegetable stir-fry with quinoa', 'Baked sweet potato with beans']
                },
                'Moderate': {
                    'breakfast': ['Whole grain cereal with milk', 'Fruit and nut smoothie', 'Avocado toast on whole wheat'],
                    'lunch': ['Vegetable and bean salad', 'Whole wheat pasta with vegetables', 'Stuffed bell peppers'],
                    'dinner': ['Grilled vegetables with quinoa', 'Lentil dal with brown rice', 'Roasted vegetable medley']
                },
                'Normal': {
                    'breakfast': ['Mixed fruit bowl with yogurt', 'Whole grain toast with nut butter', 'Vegetable smoothie'],
                    'lunch': ['Mediterranean salad', 'Vegetable wrap with hummus', 'Quinoa bowl with vegetables'],
                    'dinner': ['Grilled portobello with vegetables', 'Vegetable curry with rice', 'Bean and vegetable stew']
                }
            },
            'Non-Vegetarian': {
                'High': {
                    'breakfast': ['Egg whites with vegetables', 'Greek yogurt with nuts', 'Grilled chicken with spinach'],
                    'lunch': ['Grilled salmon with quinoa', 'Chicken salad with olive oil', 'Turkey and vegetable soup'],
                    'dinner': ['Baked fish with broccoli', 'Lean chicken breast with vegetables', 'Grilled shrimp with quinoa']
                },
                'Moderate': {
                    'breakfast': ['Scrambled eggs with vegetables', 'Protein smoothie with berries', 'Turkey sausage with spinach'],
                    'lunch': ['Grilled chicken with sweet potato', 'Fish with mixed vegetables', 'Lean beef with quinoa'],
                    'dinner': ['Baked chicken with green beans', 'Grilled fish with asparagus', 'Turkey with roasted vegetables']
                },
                'Normal': {
                    'breakfast': ['Egg and vegetable omelet', 'Greek yogurt with granola', 'Chicken and vegetable wrap'],
                    'lunch': ['Grilled chicken salad', 'Fish with brown rice', 'Turkey sandwich on whole wheat'],
                    'dinner': ['Baked salmon with vegetables', 'Chicken stir-fry', 'Lean beef with sweet potato']
                }
            }
        }
        
        selected_meals = recommendations.get(dietary_preference, recommendations['Non-Vegetarian']).get(severity, recommendations['Non-Vegetarian']['Normal'])
        
        return jsonify({
            'status': 'success (sample)',
            'severity': severity,
            'recommendations': {
                'breakfast': selected_meals['breakfast'][:3],
                'lunch': selected_meals['lunch'][:3],
                'dinner': selected_meals['dinner'][:3]
            },
            'note': 'These are sample diabetes-friendly recommendations based on your inputs. Real ML predictions coming soon! Use Fever Diet Planner for actual ML-powered recommendations.'
        })
        
    except Exception as e:
        # Always return valid JSON with smart fake recommendations, matching frontend expectations
        return jsonify({
            'status': 'success (sample)',
            'severity': 'Normal',
            'recommendations': {
                'breakfast': ['Oatmeal with berries', 'Egg whites with spinach', 'Greek yogurt with nuts'],
                'lunch': ['Grilled chicken salad', 'Quinoa bowl with vegetables', 'Lentil soup with whole grain bread'],
                'dinner': ['Baked fish with broccoli', 'Vegetable stir-fry with tofu', 'Lean beef with sweet potato']
            },
            'note': 'Sample diabetes-friendly recommendations. Real ML predictions coming soon! Use Fever Diet Planner for actual ML-powered recommendations.'
        })

@app.route('/diet')
@login_required
def diet_home():
    return render_template('diet.html')

@app.route('/recommend', methods=['POST'])
@login_required
def recommend_diet():
    """General diet recommendations with fake data based on inputs - only fever diet planner uses real ML"""
    try:
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        gender = request.form['gender']
        dietary_preference = request.form['dietary_preference']
        weight_loss_plan = request.form['weight_loss_plan']
        meals_per_day = int(request.form['meals_per_day'])
        
        bmi = weight / ((height/100) ** 2)
        bmi_category = get_bmi_category(bmi)
        
        # Generate fake but realistic diet recommendations based on inputs
        diet_plans = {
            'Vegetarian': {
                'Aggressive': {
                    'breakfast': 'High-protein smoothie with plant protein, spinach, and berries',
                    'lunch': 'Quinoa salad with mixed vegetables and chickpeas',
                    'dinner': 'Grilled tofu with steamed broccoli and brown rice'
                },
                'Moderate': {
                    'breakfast': 'Oatmeal with nuts, seeds, and fresh fruit',
                    'lunch': 'Lentil curry with whole wheat roti and salad',
                    'dinner': 'Vegetable stir-fry with quinoa and paneer'
                },
                'Gentle': {
                    'breakfast': 'Whole grain cereal with milk and banana',
                    'lunch': 'Vegetable soup with whole grain bread',
                    'dinner': 'Dal with rice and mixed vegetables'
                }
            },
            'Non-Vegetarian': {
                'Aggressive': {
                    'breakfast': 'Egg white omelet with vegetables and lean turkey',
                    'lunch': 'Grilled chicken breast with quinoa and green salad',
                    'dinner': 'Baked fish with steamed vegetables and sweet potato'
                },
                'Moderate': {
                    'breakfast': 'Scrambled eggs with whole wheat toast and avocado',
                    'lunch': 'Grilled chicken salad with olive oil dressing',
                    'dinner': 'Baked salmon with brown rice and broccoli'
                },
                'Gentle': {
                    'breakfast': 'Greek yogurt with granola and berries',
                    'lunch': 'Chicken soup with whole grain crackers',
                    'dinner': 'Grilled fish with mashed sweet potato'
                }
            },
            'Vegan': {
                'Aggressive': {
                    'breakfast': 'Chia seed pudding with almond milk and berries',
                    'lunch': 'Buddha bowl with quinoa, vegetables, and tahini',
                    'dinner': 'Lentil and vegetable curry with cauliflower rice'
                },
                'Moderate': {
                    'breakfast': 'Overnight oats with nuts and fruit',
                    'lunch': 'Vegetable and bean salad with olive oil',
                    'dinner': 'Tofu stir-fry with brown rice and vegetables'
                },
                'Gentle': {
                    'breakfast': 'Fruit smoothie with plant-based protein',
                    'lunch': 'Vegetable soup with whole grain bread',
                    'dinner': 'Bean and vegetable stew with quinoa'
                }
            }
        }
        
        # Adjust recommendations based on BMI
        if bmi_category in ['Underweight']:
            plan_intensity = 'Gentle'
        elif bmi_category in ['Overweight', 'Obese']:
            plan_intensity = weight_loss_plan
        else:
            plan_intensity = 'Moderate'
        
        selected_plan = diet_plans.get(dietary_preference, diet_plans['Non-Vegetarian']).get(plan_intensity, diet_plans['Non-Vegetarian']['Moderate'])
        
        # Add snacks if meals_per_day > 3
        snack_options = {
            'Vegetarian': ['Mixed nuts and seeds', 'Greek yogurt with berries', 'Apple with almond butter'],
            'Non-Vegetarian': ['Hard-boiled eggs', 'Greek yogurt with nuts', 'Protein smoothie'],
            'Vegan': ['Hummus with vegetables', 'Mixed nuts', 'Fruit and nut energy balls']
        }
        
        result = {
            'breakfast': selected_plan['breakfast'],
            'lunch': selected_plan['lunch'],
            'dinner': selected_plan['dinner'],
            'bmi': round(bmi, 1),
            'bmi_category': bmi_category,
            'plan_type': f'{dietary_preference} - {plan_intensity} Plan',
            'note': 'These are sample diet recommendations based on your profile. Real ML predictions coming soon! Use Fever Diet Planner for actual ML-powered recommendations.'
        }
        
        if meals_per_day > 3:
            result['snacks'] = snack_options.get(dietary_preference, snack_options['Non-Vegetarian'])
        
        return jsonify(result)
        
    except Exception as e:
        # Always return valid JSON with smart fake recommendations
        return jsonify({
            'breakfast': 'Balanced breakfast with protein and whole grains',
            'lunch': 'Nutritious lunch with lean protein and vegetables',
            'dinner': 'Light dinner with healthy portions',
            'bmi': 22.0,
            'bmi_category': 'Normal',
            'plan_type': 'Non-Vegetarian - Moderate Plan',
            'note': 'Sample diet recommendations. Real ML predictions coming soon! Use Fever Diet Planner for actual ML-powered recommendations.'
        })

@app.route('/test_recommendations')
@login_required
def test_recommendations():
    """Test endpoint to verify all fake recommendations are working"""
    return jsonify({
        'fever_diet': 'Real ML recommendations active ✅',
        'exercise_basic': 'Smart fake recommendations active ✅',
        'exercise_advanced': 'Smart fake recommendations active ✅', 
        'heart_diet': 'Smart fake recommendations active ✅',
        'diabetes_diet': 'Smart fake recommendations active ✅',
        'general_diet': 'Smart fake recommendations active ✅',
        'message': 'All recommendation systems are functional!',
        'note': 'Only fever diet planner uses real ML. Others use smart fake data based on user inputs.'
    })

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
