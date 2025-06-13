import unittest
import mysql.connector
import bcrypt
import logic.auth 

TEST_DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "pdf_brainbox_test"
}

class TestAuthDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.conn = mysql.connector.connect(**TEST_DB_CONFIG)
        cls.cursor = cls.conn.cursor()
        cls.cursor.execute("DROP TABLE IF EXISTS test_users")
        cls.cursor.execute("""
            CREATE TABLE test_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        """)
        cls.conn.commit()
        print("\n Setup test table: test_users")

    @classmethod
    def tearDownClass(cls):
        cls.cursor.execute("DROP TABLE IF EXISTS test_users")
        cls.conn.commit()
        cls.cursor.close()
        cls.conn.close()
        print(" Dropped test table and closed DB connection.")

    def test_user_creation_and_verification(self):
        # Test new user creation
        username = "testuser"
        password = "securepassword123"
        
        print("\n Testing user creation...")
        result = logic.auth.create_user(username, password)
        print(f"Created user '{username}':", result)
        self.assertTrue(result)

        print("\n Testing duplicate user creation...")
        duplicate_result = logic.auth.create_user(username, password)
        print(f"Duplicate user creation (should be False):", duplicate_result)
        self.assertFalse(duplicate_result)

        print("\n Testing correct password verification...")
        verified = logic.auth.verify_user(username, password)
        print(f"Login with correct password:", verified)
        self.assertTrue(verified)

        print("\n Testing wrong password...")
        wrong_password_result = logic.auth.verify_user(username, "wrongpassword")
        print(f"Login with wrong password:", wrong_password_result)
        self.assertFalse(wrong_password_result)

        print("\n Testing non-existent user...")
        nonexistent_user = logic.auth.verify_user("ghostuser", "anypass")
        print(f"Login with non-existent user:", nonexistent_user)
        self.assertFalse(nonexistent_user)
if __name__ == "__main__":
    unittest.main()
