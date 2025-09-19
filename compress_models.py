"""
Model Compression Script for FitIntel Fever Diet Planner
Compresses ML models to fit within Render's 512MB free tier limit
"""

import os
import joblib
import pickle
import gzip
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

def get_model_size(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0

def compress_with_gzip(input_file, output_file):
    """Compress model file using gzip"""
    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            f_out.write(f_in.read())
    return get_model_size(output_file)

def simplify_random_forest(model, max_estimators=10, max_depth=5):
    """Simplify RandomForest by reducing trees and depth"""
    if hasattr(model, 'estimators_'):
        # Keep only first few estimators
        model.estimators_ = model.estimators_[:max_estimators]
        model.n_estimators = len(model.estimators_)
        
        # Reduce tree depth
        for estimator in model.estimators_:
            if hasattr(estimator, 'tree_'):
                # This is a simplified approach - in practice, you'd retrain
                pass
    return model

def create_compressed_fever_models():
    """Create compressed versions of fever models"""
    
    models_dir = 'models'
    compressed_dir = 'models_compressed'
    
    # Create compressed models directory
    if not os.path.exists(compressed_dir):
        os.makedirs(compressed_dir)
    
    print("🔄 Starting model compression process...")
    print(f"📁 Original models directory: {models_dir}")
    print(f"📁 Compressed models directory: {compressed_dir}")
    
    compression_results = {}
    
    # List of fever models to compress
    fever_models = [
        'breakfast_model_fever.pkl',
        'lunch_model_fever.pkl', 
        'dinner_model_fever.pkl',
        'meal_encoders_fever.pkl',
        'gender_encoder_fever.pkl'
    ]
    
    for model_file in fever_models:
        input_path = os.path.join(models_dir, model_file)
        
        if not os.path.exists(input_path):
            print(f"⚠️ Model not found: {input_path}")
            continue
            
        original_size = get_model_size(input_path)
        print(f"\n📊 Processing: {model_file}")
        print(f"   Original size: {original_size:.2f} MB")
        
        try:
            # Load the model
            model = joblib.load(input_path)
            
            # Apply model-specific compression
            if 'breakfast_model' in model_file or 'lunch_model' in model_file or 'dinner_model' in model_file:
                # These are the large models - apply aggressive compression
                if hasattr(model, 'estimators_'):
                    print(f"   📉 Original estimators: {len(model.estimators_)}")
                    # Reduce to 5 estimators for massive size reduction
                    model.estimators_ = model.estimators_[:5]
                    model.n_estimators = 5
                    print(f"   📉 Compressed estimators: {len(model.estimators_)}")
                    
                # Try to simplify further if it's a complex ensemble
                if hasattr(model, 'max_depth'):
                    model.max_depth = 3
                    
            elif 'encoders' in model_file or 'gender_encoder' in model_file:
                # These are small encoders - minimal compression needed
                pass
            
            # Save compressed model (uncompressed first)
            temp_path = os.path.join(compressed_dir, model_file)
            joblib.dump(model, temp_path)
            compressed_size = get_model_size(temp_path)
            
            # Apply gzip compression
            gzip_path = temp_path + '.gz'
            final_size = compress_with_gzip(temp_path, gzip_path)
            
            # Keep the smaller version
            if final_size < compressed_size:
                os.remove(temp_path)
                # Rename .gz to .pkl for consistent loading
                final_path = temp_path
                os.rename(gzip_path, final_path)
                actual_size = final_size
                compression_method = "joblib + gzip"
            else:
                os.remove(gzip_path)
                actual_size = compressed_size
                compression_method = "joblib only"
            
            reduction = ((original_size - actual_size) / original_size) * 100
            
            print(f"   ✅ Compressed size: {actual_size:.2f} MB")
            print(f"   📈 Size reduction: {reduction:.1f}%")
            print(f"   🔧 Method: {compression_method}")
            
            compression_results[model_file] = {
                'original_size': original_size,
                'compressed_size': actual_size,
                'reduction_percent': reduction,
                'method': compression_method
            }
            
        except Exception as e:
            print(f"   ❌ Error compressing {model_file}: {str(e)}")
            compression_results[model_file] = {
                'error': str(e)
            }
    
    # Calculate total sizes
    total_original = sum([r.get('original_size', 0) for r in compression_results.values()])
    total_compressed = sum([r.get('compressed_size', 0) for r in compression_results.values()])
    total_reduction = ((total_original - total_compressed) / total_original) * 100 if total_original > 0 else 0
    
    print(f"\n📈 COMPRESSION SUMMARY:")
    print(f"   📊 Total original size: {total_original:.2f} MB")
    print(f"   📊 Total compressed size: {total_compressed:.2f} MB") 
    print(f"   📊 Total reduction: {total_reduction:.1f}%")
    print(f"   🎯 Target (512MB): {'✅ FITS' if total_compressed < 400 else '❌ TOO LARGE'}")
    
    # Save compression report
    with open(os.path.join(compressed_dir, 'compression_report.json'), 'w') as f:
        json.dump({
            'compression_results': compression_results,
            'summary': {
                'total_original_mb': total_original,
                'total_compressed_mb': total_compressed,
                'total_reduction_percent': total_reduction,
                'fits_in_512mb': total_compressed < 400  # Leave some headroom
            }
        }, f, indent=2)
    
    return compression_results, total_compressed

if __name__ == "__main__":
    results, total_size = create_compressed_fever_models()
    
    if total_size < 400:  # 400MB gives us headroom for the app itself
        print(f"\n🎉 SUCCESS! Compressed models fit in free tier!")
        print(f"💡 Next step: Update main.py to use compressed models")
    else:
        print(f"\n⚠️ Still too large. Consider further optimizations:")
        print(f"   - Use even fewer estimators (2-3)")
        print(f"   - Implement rule-based fallbacks")
        print(f"   - Use external model storage")