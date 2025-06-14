import pyrebase

firebase_config = {
    "apiKey": "<API_KEY>",
    "authDomain": "<PROJECT_ID>.firebaseapp.com",
    "databaseURL": "https://<PROJECT_ID>.firebaseio.com",
    "projectId": "<PROJECT_ID>",
    "storageBucket": "<PROJECT_ID>.appspot.com",
    "messagingSenderId": "<SENDER_ID>",
    "appId": "<APP_ID>"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

def create_user(username, password):
    try:
        auth.create_user_with_email_and_password(username, password)
        return True
    except:
        return False

def verify_user(username, password):
    try:
        user = auth.sign_in_with_email_and_password(username, password)
        return True
    except:
        return False
