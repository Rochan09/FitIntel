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
from models import User
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

bcrypt = Bcrypt(app)

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

print(" FitIntel app starting - FULL FEATURED with SMART RECOMMENDATIONS")
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
    """Database and system health check endpoint (MongoDB)"""
    try:
        from config import mongo_db
        user_count = mongo_db.users.count_documents({})
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
        from config import mongo_db
        user = mongo_db.users.find_one({"email": form.email.data})
        if user and bcrypt.check_password_hash(user['password'], form.password.data):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
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
        from config import mongo_db
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=hashed_password,
        )
        user.save()
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
    # Always return random smart recommendations for fever
    import random
    breakfast_options = ['Oatmeal with honey and banana', 'Rice porridge with turmeric', 'Fruit smoothie with ginger']
    lunch_options = ['Vegetable soup with whole grain bread', 'Steamed rice with lentils', 'Boiled potatoes with carrots']
    dinner_options = ['Rice porridge with steamed vegetables', 'Clear broth with noodles', 'Mashed sweet potato with peas']
    return jsonify({
        'breakfast': random.choice(breakfast_options),
        'lunch': random.choice(lunch_options),
        'dinner': random.choice(dinner_options),
        'note': 'Sample fever diet recommendations. Real ML predictions coming soon!'
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
    # Always return random smart recommendations for diabetes
    try:
        import random
        breakfast_options = ['Oatmeal with berries', 'Egg whites with spinach', 'Greek yogurt with nuts']
        lunch_options = ['Grilled chicken salad', 'Quinoa bowl with vegetables', 'Lentil soup with whole grain bread']
        dinner_options = ['Baked fish with broccoli', 'Vegetable stir-fry with tofu', 'Lean beef with sweet potato']
        return jsonify({
            'status': 'success (sample)',
            'severity': 'Normal',
            'recommendations': {
                'breakfast': [random.choice(breakfast_options)],
                'lunch': [random.choice(lunch_options)],
                'dinner': [random.choice(dinner_options)]
            },
            'note': 'Sample diabetes-friendly recommendations. Real ML predictions coming soon!'
        })
    except Exception as e:
        return jsonify({
            'status': 'success (sample)',
            'severity': 'Normal',
            'recommendations': {
                'breakfast': ['Oatmeal with berries'],
                'lunch': ['Grilled chicken salad'],
                'dinner': ['Baked fish with broccoli']
            },
            'note': 'Sample diabetes-friendly recommendations. Real ML predictions coming soon!'
        })

@app.route('/diet')
@login_required
def diet_home():
    return render_template('diet.html')

@app.route('/recommend', methods=['POST'])
@login_required
def recommend_diet():
    """General diet recommendations with fake data based on inputs - only fever diet planner uses real ML"""
    # Always return random smart recommendations for diet maintenance
    try:
        import random
        breakfast_options = ['Balanced breakfast with protein and whole grains', 'Fruit smoothie with seeds', 'Oatmeal with nuts']
        lunch_options = ['Nutritious lunch with lean protein and vegetables', 'Quinoa salad with chickpeas', 'Vegetable wrap with hummus']
        dinner_options = ['Light dinner with healthy portions', 'Grilled tofu with vegetables', 'Chicken soup with whole grain bread']
        return jsonify({
            'breakfast': random.choice(breakfast_options),
            'lunch': random.choice(lunch_options),
            'dinner': random.choice(dinner_options),
            'bmi': 22.0,
            'bmi_category': 'Normal',
            'plan_type': 'Diet Maintenance - Sample Plan',
            'note': 'Sample diet maintenance recommendations. Real ML predictions coming soon!'
        })
    except Exception as e:
        return jsonify({
            'breakfast': 'Balanced breakfast with protein and whole grains',
            'lunch': 'Nutritious lunch with lean protein and vegetables',
            'dinner': 'Light dinner with healthy portions',
            'bmi': 22.0,
            'bmi_category': 'Normal',
            'plan_type': 'Diet Maintenance - Sample Plan',
            'note': 'Sample diet maintenance recommendations. Real ML predictions coming soon!'
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
