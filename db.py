import pyrebase
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

# Pyrebase config from static keys (you must also add these to Streamlit secrets)
firebase_config = {
    "apiKey": st.secrets["pyrebase"]["apiKey"],
    "authDomain": st.secrets["pyrebase"]["authDomain"],
    "databaseURL": st.secrets["pyrebase"]["databaseURL"],
    "projectId": st.secrets["pyrebase"]["projectId"],
    "storageBucket": st.secrets["pyrebase"]["storageBucket"],
    "messagingSenderId": st.secrets["pyrebase"]["messagingSenderId"],
    "appId": st.secrets["pyrebase"]["appId"]
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
realtime_db = firebase.database()

# Firestore init
firebase_credentials = {
    "type": st.secrets.firebase.type,
    "project_id": st.secrets.firebase.project_id,
    "private_key_id": st.secrets.firebase.private_key_id,
    "private_key": st.secrets.firebase.private_key.replace("\\n", "\n"),
    "client_email": st.secrets.firebase.client_email,
    "client_id": st.secrets.firebase.client_id,
    "auth_uri": st.secrets.firebase.auth_uri,
    "token_uri": st.secrets.firebase.token_uri,
    "auth_provider_x509_cert_url": st.secrets.firebase.auth_provider_x509_cert_url,
    "client_x509_cert_url": st.secrets.firebase.client_x509_cert_url,
    "universe_domain": st.secrets.firebase.universe_domain
}

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)

firestore_db = firestore.client()


def create_user(email, password):
    try:
        auth.create_user_with_email_and_password(email, password)
        return True
    except Exception as e:
        print(e)
        return False


def verify_user(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return user  # contains user['idToken'] etc.
    except Exception as e:
        print(e)
        return None


def save_note_to_firestore(user_email, note_text):
    firestore_db.collection("notes").add({
        "user": user_email,
        "note": note_text
    })


def get_notes_for_user(user_email):
    docs = firestore_db.collection("notes").where("user", "==", user_email).stream()
    return [doc.to_dict() for doc in docs]
