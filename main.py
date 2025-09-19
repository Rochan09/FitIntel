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
app.config['SECRET_KEY'] = 'ASLDFJASDLFADS'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:rochan@localhost:5432/flask_auth_db'

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
def load_exercise_models():
    """Load exercise recommendation models"""
    print("Loading exercise models...")
    
    models = {}
    for i in range(1, 4):
        model_path = f'models/model_Exercise_{i}.pkl'
        if os.path.exists(model_path):
            try:
                models[f'Exercise_{i}'] = joblib.load(model_path)
            except (AttributeError, pickle.UnpicklingError, FileNotFoundError) as e:
                print(f"Warning: Could not load model {model_path}. This may be due to a version mismatch in scikit-learn or a missing model file. Error: {e}")

    models1 = {}
    for i in range(1, 4):
        model_path = f'models/model1_Exercise_{i}.pkl'
        if os.path.exists(model_path):
            try:
                models1[f'Exercise_{i}'] = joblib.load(model_path)
            except (AttributeError, pickle.UnpicklingError, FileNotFoundError) as e:
                print(f"Warning: Could not load model {model_path}. This may be due to a version mismatch in scikit-learn or a missing model file. Error: {e}")
    
    unique_exercises = []
    file_path = 'models/unique_exercises.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            unique_exercises = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(models)} exercise models from first set")
    print(f"Loaded {len(models1)} exercise models from second set")
    print(f"Loaded {len(unique_exercises)} unique exercises")
    
    return models, models1, unique_exercises

def load_food_models():
    """Load food recommendation models"""
    print("Loading food models...")
    
    food_models = {
        'fever': {},
        'heart': {},
        'diabetes': {},
        'diet': {}
    }
    
    try:
        food_models['fever']['breakfast_model'] = joblib.load('models/breakfast_model_fever.pkl')
        food_models['fever']['lunch_model'] = joblib.load('models/lunch_model_fever.pkl')
        food_models['fever']['dinner_model'] = joblib.load('models/dinner_model_fever.pkl')
        food_models['fever']['meal_encoders'] = joblib.load('models/meal_encoders_fever.pkl')
        food_models['fever']['gender_encoder'] = joblib.load('models/gender_encoder_fever.pkl')
        print("Loaded fever food models")
    except (AttributeError, pickle.UnpicklingError, FileNotFoundError, MemoryError) as e:
        print(f"Warning: Could not load fever food models. This may be due to a version mismatch, a corrupted model file, or a memory allocation issue. Error: {e}")
    
    try:
        with open('models/risk_model_heart.pkl', 'rb') as f:
            food_models['heart']['risk_model'] = pickle.load(f)
        with open('models/scaler_heart.pkl', 'rb') as f:
            food_models['heart']['scaler'] = pickle.load(f)
        with open('models/encoders_heart.pkl', 'rb') as f:
            food_models['heart']['encoders'] = pickle.load(f)
        with open('models/meal_recommendations_heart.pkl', 'rb') as f:
            food_models['heart']['meal_recommendations'] = pickle.load(f)
        print("Loaded heart food models")
    except (AttributeError, pickle.UnpicklingError, FileNotFoundError, MemoryError) as e:
        print(f"Warning: Could not load heart food models. This may be due to a version mismatch, a corrupted model file, or a memory allocation issue. Error: {e}")
    
    try:
        food_models['diabetes']['breakfast_model'] = joblib.load('models/diabetes_food_breakfast_model.pkl')
        food_models['diabetes']['lunch_model'] = joblib.load('models/diabetes_food_lunch_model.pkl')
        food_models['diabetes']['dinner_model'] = joblib.load('models/diabetes_food_dinner_model.pkl')
        food_models['diabetes']['breakfast_vectorizer'] = joblib.load('models/diabetes_food_breakfast_vectorizer.pkl')
        food_models['diabetes']['lunch_vectorizer'] = joblib.load('models/diabetes_food_lunch_vectorizer.pkl')
        food_models['diabetes']['dinner_vectorizer'] = joblib.load('models/diabetes_food_dinner_vectorizer.pkl')
        print("Loaded diabetes food models")
    except (AttributeError, pickle.UnpicklingError, FileNotFoundError, MemoryError) as e:
        print(f"Warning: Could not load diabetes food models. This may be due to a version mismatch, a corrupted model file, or a memory allocation issue. Error: {e}")
    
    try:
        food_models['diet']['breakfast_model'] = joblib.load('models/breakfast_model.pkl')
        food_models['diet']['lunch_model'] = joblib.load('models/lunch_model.pkl')
        food_models['diet']['dinner_model'] = joblib.load('models/dinner_model.pkl')
        food_models['diet']['scaler'] = joblib.load('models/scaler.pkl')
        food_models['diet']['le_gender'] = joblib.load('models/le_gender.pkl')
        food_models['diet']['le_dietary_pref'] = joblib.load('models/le_dietary_pref.pkl')
        food_models['diet']['le_weight_loss_plan'] = joblib.load('models/le_weight_loss_plan.pkl')
        print("Loaded diet food models")
    except (AttributeError, pickle.UnpicklingError, FileNotFoundError, MemoryError) as e:
        print(f"Warning: Could not load diet food models. This may be due to a version mismatch, a corrupted model file, or a memory allocation issue. Error: {e}")
    
    return food_models

# Load all models at startup
with app.app_context():
    db.create_all()
    exercise_models, exercise_models1, unique_exercises = load_exercise_models()
    food_models = load_food_models()

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
    """Predict food recommendations for a new diabetes patient"""
    new_sample = pd.DataFrame({
        'age': [age],
        'weight': [weight],
        'gender': [gender],
        'fasting_blood_sugar': [fasting_blood_sugar],
        'hba1c': [hba1c],
        'diabetes_type': [diabetes_type],
        'dietary_preference': [dietary_preference]
    })
    
    try:
        breakfast_model = food_models['diabetes']['breakfast_model']
        breakfast_vectorizer = food_models['diabetes']['breakfast_vectorizer']
        lunch_model = food_models['diabetes']['lunch_model']
        lunch_vectorizer = food_models['diabetes']['lunch_vectorizer']
        dinner_model = food_models['diabetes']['dinner_model']
        dinner_vectorizer = food_models['diabetes']['dinner_vectorizer']
        
        breakfast_categories = breakfast_model.predict(new_sample)[0]
        lunch_categories = lunch_model.predict(new_sample)[0]
        dinner_categories = dinner_model.predict(new_sample)[0]
        
        breakfast_items = map_categories_to_food_items(breakfast_categories, 'breakfast', dietary_preference)
        lunch_items = map_categories_to_food_items(lunch_categories, 'lunch', dietary_preference)
        dinner_items = map_categories_to_food_items(dinner_categories, 'dinner', dietary_preference)
    except (AttributeError, pickle.UnpicklingError, FileNotFoundError) as e:
        print(f"Error in diabetes prediction: {e}. This might be due to a library version mismatch or missing model files.")
        breakfast_items = map_categories_to_food_items("", 'breakfast', dietary_preference)
        lunch_items = map_categories_to_food_items("", 'lunch', dietary_preference)
        dinner_items = map_categories_to_food_items("", 'dinner', dietary_preference)
        
    return {
        'breakfast': breakfast_items,
        'lunch': lunch_items,
        'dinner': dinner_items
    }

# ==================== Authentication Routes ====================
@app.route('/')
def home():
    return render_template('index.html')

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
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        fitness_level = int(request.form['fitness_level'])
        health_conditions = request.form.getlist('health_conditions')
        time_available = int(request.form['time_available'])
        equipment_access = request.form['equipment_access']
        exercise_preference = request.form['exercise_preference']
        intensity_preference = request.form['intensity_preference']
        
        health_conditions_str = ', '.join(health_conditions) if health_conditions else 'None'
        bmi = weight / ((height/100) ** 2)
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Weight_kg': [weight],
            'Height_cm': [height],
            'BMI': [bmi],
            'Fitness_Level': [fitness_level],
            'Health_Conditions': [health_conditions_str],
            'Time_Available_Min': [time_available],
            'Equipment_Access': [equipment_access],
            'Exercise_Preference': [exercise_preference],
            'Intensity_Preference': [intensity_preference]
        })
        
        recommended_exercises = []
        for i in range(1, 4):
            if f'Exercise_{i}' in exercise_models:
                prediction = exercise_models[f'Exercise_{i}'].predict(input_data)
                if prediction[0] not in recommended_exercises:
                    recommended_exercises.append(prediction[0])
                    
        while len(recommended_exercises) < 2 and unique_exercises:
            popular_exercise = unique_exercises[0]
            if popular_exercise not in recommended_exercises:
                recommended_exercises.append(popular_exercise)
        
        return jsonify({
            'success': True,
            'exercises': recommended_exercises,
            'profile': {
                'age': age,
                'gender': gender,
                'bmi': round(bmi, 1),
                'fitness_level': fitness_level
            }
        })
        
    except Exception as e:
        print(f"Error in predict1: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        fitness_level = int(request.form['fitness_level'])
        weight_loss_goal = int(request.form['weight_loss_goal'])
        health_conditions = request.form.getlist('health_conditions')
        previous_injuries = request.form['previous_injuries']
        time_available = int(request.form['time_available'])
        equipment_access = request.form['equipment_access']
        exercise_preference = request.form['exercise_preference']
        intensity_preference = request.form['intensity_preference']
        target_areas = request.form.getlist('target_areas')
        
        health_conditions_str = ', '.join(health_conditions) if health_conditions else 'None'
        target_areas_str = ', '.join(target_areas) if target_areas else 'Full body'
        bmi = weight / ((height/100) ** 2)
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Weight_kg': [weight],
            'Height_cm': [height],
            'BMI': [bmi],
            'Fitness_Level': [fitness_level],
            'Weight_Loss_Goal_kg': [weight_loss_goal],
            'Health_Conditions': [health_conditions_str],
            'Previous_Injuries': [previous_injuries],
            'Time_Available_Min': [time_available],
            'Equipment_Access': [equipment_access],
            'Exercise_Preference': [exercise_preference],
            'Intensity_Preference': [intensity_preference],
            'Target_Body_Areas': [target_areas_str]
        })
        
        recommended_exercises = []
        for i in range(1, 4):
            if f'Exercise_{i}' in exercise_models1:
                prediction = exercise_models1[f'Exercise_{i}'].predict(input_data)
                if prediction[0] not in recommended_exercises:
                    recommended_exercises.append(prediction[0])
        
        while len(recommended_exercises) < 2 and unique_exercises:
            popular_exercises = unique_exercises[:5]
            for exercise in popular_exercises:
                if exercise not in recommended_exercises:
                    recommended_exercises.append(exercise)
                    break
        
        exercise_details = []
        for exercise in recommended_exercises:
            duration = min(time_available // len(recommended_exercises), 20) if time_available > 30 else min(time_available // len(recommended_exercises), 15)
            if intensity_preference == 'High':
                duration = max(duration - 5, 10)
            elif intensity_preference == 'Low':
                duration = min(duration + 5, 30)
            
            if 'HIIT' in exercise or 'Burpees' in exercise:
                description = "High-intensity exercise that burns calories quickly and improves cardiovascular fitness."
            elif any(term in exercise.lower() for term in ['walking', 'jogging', 'running', 'cycling', 'swimming']):
                description = "Excellent cardio exercise for burning calories and improving endurance."
            elif any(term in exercise.lower() for term in ['squats', 'press', 'lunges', 'push-ups', 'pull-ups']):
                description = "Strength exercise that builds muscle, which helps increase metabolic rate."
            elif any(term in exercise.lower() for term in ['yoga', 'pilates', 'stretching', 'tai chi']):
                description = "Helps with flexibility, stress reduction, and core strength."
            else:
                description = "Effective exercise for overall fitness and calorie burning."
            
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
            }
        })
        
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# ==================== Food Recommendation Routes ====================
@app.route('/fever')
@login_required
def fever_home():
    return render_template('fever.html')

@app.route('/predict_fever', methods=['POST'])
@login_required
def predict_fever():
    if 'fever' not in food_models or not all(key in food_models['fever'] for key in ['breakfast_model', 'lunch_model', 'dinner_model']):
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

    input_data['gender'] = food_models['fever']['gender_encoder'].transform(input_data['gender'])

    breakfast_pred = food_models['fever']['breakfast_model'].predict(input_data)[0]
    lunch_pred = food_models['fever']['lunch_model'].predict(input_data)[0]
    dinner_pred = food_models['fever']['dinner_model'].predict(input_data)[0]

    breakfast_rec = food_models['fever']['meal_encoders']['breakfast'].inverse_transform([breakfast_pred])[0]
    lunch_rec = food_models['fever']['meal_encoders']['lunch'].inverse_transform([lunch_pred])[0]
    dinner_rec = food_models['fever']['meal_encoders']['dinner'].inverse_transform([dinner_pred])[0]

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
    if 'heart' not in food_models or not all(key in food_models['heart'] for key in ['risk_model', 'scaler', 'encoders']):
        return jsonify({'error': 'Heart models not loaded'}), 500
    
    age = int(request.form['age'])
    weight = int(request.form['weight'])
    gender = request.form['gender']
    cholesterol = int(request.form['cholesterol'])
    bp_systolic = int(request.form['bp_systolic'])  
    bp_diastolic = int(request.form['bp_diastolic'])
    obesity = request.form['obesity']
    
    gender_encoded = food_models['heart']['encoders']['gender'].transform([gender])[0]
    obesity_encoded = food_models['heart']['encoders']['obesity'].transform([obesity])[0]
    
    input_data = np.array([[age, weight, gender_encoded, cholesterol, bp_systolic, bp_diastolic, obesity_encoded]])
    input_data_scaled = food_models['heart']['scaler'].transform(input_data)
    risk_category = food_models['heart']['risk_model'].predict(input_data_scaled)[0]
    
    category_recommendations = food_models['heart']['meal_recommendations'][risk_category]
    breakfast_options = random.choice(category_recommendations['breakfast'])
    lunch_options = random.choice(category_recommendations['lunch'])
    dinner_options = random.choice(category_recommendations['dinner'])
    
    result = {
        'risk_category': risk_category,
        'breakfast_recommendation': ', '.join(breakfast_options),
        'lunch_recommendation': ', '.join(lunch_options),
        'dinner_recommendation': ', '.join(dinner_options)
    }
    
    return render_template('heart.html', prediction=result)

@app.route('/diabetes')
@login_required
def diabetes_home():
    return render_template('diabetes.html')

@app.route('/predict_diabetes', methods=['POST'])
@login_required
def predict_diabetes():
    data = request.form
    
    try:
        age = int(data.get('age'))
        weight = float(data.get('weight'))
        gender = data.get('gender')
        fasting_blood_sugar = float(data.get('fasting_blood_sugar'))
        hba1c = float(data.get('hba1c'))
        diabetes_type = data.get('diabetes_type')
        dietary_preference = data.get('dietary_preference')
        
        recommendations = predict_food_recommendations(
            age, weight, gender, fasting_blood_sugar, 
            hba1c, diabetes_type, dietary_preference
        )
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/diet')
@login_required
def diet_home():
    return render_template('diet.html')

@app.route('/recommend', methods=['POST'])
@login_required
def recommend_diet():
    age = int(request.form['age'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    gender = request.form['gender']
    dietary_preference = request.form['dietary_preference']
    weight_loss_plan = request.form['weight_loss_plan']
    meals_per_day = int(request.form['meals_per_day'])

    gender_encoded = food_models['diet']['le_gender'].transform([gender])[0]
    dietary_pref_encoded = food_models['diet']['le_dietary_pref'].transform([dietary_preference])[0]
    weight_loss_plan_encoded = food_models['diet']['le_weight_loss_plan'].transform([weight_loss_plan])[0]

    input_features = np.array([
        age, height, weight, gender_encoded, 
        dietary_pref_encoded, weight_loss_plan_encoded, meals_per_day
    ]).reshape(1, -1)

    input_scaled = food_models['diet']['scaler'].transform(input_features)

    breakfast_pred = food_models['diet']['breakfast_model'].predict(input_scaled)[0]
    lunch_pred = food_models['diet']['lunch_model'].predict(input_scaled)[0]
    dinner_pred = food_models['diet']['dinner_model'].predict(input_scaled)[0]

    return jsonify({
        'breakfast': breakfast_pred,
        'lunch': lunch_pred,
        'dinner': dinner_pred
    })

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
