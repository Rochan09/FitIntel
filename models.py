from config import mongo_db

class User:
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password

    def save(self):
        mongo_db.users.insert_one({
            "username": self.username,
            "email": self.email,
            "password": self.password
        })

    @staticmethod
    def find_by_username(username):
        return mongo_db.users.find_one({"username": username})
