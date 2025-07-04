import mysql.connector
import bcrypt

def get_connection():
    return mysql.connector.connect(
        host="localhost",       # or "127.0.0.1"
        user="root",
        password="1234",
        database="pdf_brainbox_test"
    )

def create_user(username, password):
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO test_users (username, password) VALUES (%s, %s)", (username, hashed_pw))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        return False
    finally:
        cursor.close()
        conn.close()

def verify_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM test_users WHERE username = %s", (username,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    if result:
        return bcrypt.checkpw(password.encode('utf-8'), result[0].encode())
    return False
