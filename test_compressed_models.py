"""
Test script to verify compressed fever models work correctly
"""

import sys
import os
sys.path.append('.')

from main import get_food_models
import pandas as pd

def test_compressed_fever_models():
    print("🧪 Testing compressed fever models...")
    
    try:
        # Load fever models
        fever_models = get_food_models('fever')
        
        if not fever_models:
            print("❌ No fever models loaded")
            return False
        
        # Check if all required models are present
        required_models = ['breakfast_model', 'lunch_model', 'dinner_model', 'meal_encoders', 'gender_encoder']
        missing_models = [m for m in required_models if m not in fever_models or fever_models[m] is None]
        
        if missing_models:
            print(f"❌ Missing models: {missing_models}")
            return False
        
        print("✅ All fever models loaded successfully")
        
        # Test prediction with sample data
        print("\n🧪 Testing fever prediction...")
        
        sample_data = pd.DataFrame({
            'age': [25],
            'weight': [70],
            'gender': ['Male'],
            'fever_level': [101.5]
        })
        
        # Transform gender using the encoder
        sample_data['gender'] = fever_models['gender_encoder'].transform(sample_data['gender'])
        
        # Make predictions
        breakfast_pred = fever_models['breakfast_model'].predict(sample_data)[0]
        lunch_pred = fever_models['lunch_model'].predict(sample_data)[0]
        dinner_pred = fever_models['dinner_model'].predict(sample_data)[0]
        
        # Decode predictions
        breakfast_rec = fever_models['meal_encoders']['breakfast'].inverse_transform([breakfast_pred])[0]
        lunch_rec = fever_models['meal_encoders']['lunch'].inverse_transform([lunch_pred])[0]
        dinner_rec = fever_models['meal_encoders']['dinner'].inverse_transform([dinner_pred])[0]
        
        print(f"🍳 Breakfast recommendation: {breakfast_rec}")
        print(f"🍽️ Lunch recommendation: {lunch_rec}")
        print(f"🍽️ Dinner recommendation: {dinner_rec}")
        
        print("\n✅ Compressed fever models are working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing compressed models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_compressed_fever_models()
    if success:
        print("\n🎉 SUCCESS: Compressed models are ready for deployment!")
    else:
        print("\n💥 FAILURE: Issues found with compressed models")
        sys.exit(1)