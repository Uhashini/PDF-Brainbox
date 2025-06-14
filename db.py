# db.py
import requests
import os

# Replace with your Firebase project’s web API key
FIREBASE_API_KEY = os.getenv("AIzaSyDYlGv0O7QmRrBEwZ3Xu8bcNNSbcK_PMTU") or "AIzaSyDYlGv0O7QmRrBEwZ3Xu8bcNNSbcK_PMTU"

# Firebase REST endpoints
SIGNUP_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
LOGIN_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"

def create_user(email, password):
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    res = requests.post(SIGNUP_URL, json=payload)
    
    if res.status_code == 200:
        return True
    else:
        try:
            error = res.json()["error"]["message"]
            print("Firebase error:", error)
        except:
            error = "UNKNOWN_ERROR"
        return error  # Return error message as string

def verify_user(email, password):
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    res = requests.post(LOGIN_URL, json=payload)

    if res.status_code == 200:
        return True
    else:
        try:
            error = res.json()["error"]["message"]
            print("Firebase error:", error)
        except:
            error = "UNKNOWN_ERROR"
        return error
