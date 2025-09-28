# Configuration settings for FitIntel

from pymongo import MongoClient

class Config:
    SECRET_KEY = '8b61e56e6e1d7b766ccd916d3ba4eb1453bf794f4edea5ee41834204257823b0'
    DEBUG = True
    # MongoDB Atlas connection string
    MONGO_URI = 'mongodb+srv://fitintel:fitintel@cluster0.botwimt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

# MongoDB client setup (global, can be imported elsewhere)
mongo_client = MongoClient(Config.MONGO_URI)
mongo_db = mongo_client['fitintel']  # Use 'fitintel' database
