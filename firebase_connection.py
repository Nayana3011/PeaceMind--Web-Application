import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase Admin SDK
cred = credentials.Certificate('serviceAccountKey.json')  # Replace with your JSON key file
try:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://stressdetection-3c1a2-default-rtdb.firebaseio.com'
    })
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error: {e}")


# Reference to the "registration" node
ref = db.reference('registration')



# Create a record
def create_user(user_id, name, phno, email, password):
    ref.child(user_id).set({
        'id': user_id,
        'name': name,
        'phno': phno,
        'email': email,
        'password': password
    })
    print(f'User {user_id} created successfully.')

# Read a record
def read_user(user_id):
    user = ref.child(user_id).get()
    if user:
        print(f'User {user_id}: {user}')
    else:
        print(f'User {user_id} not found.')

# Update a record
def update_user(user_id, updated_data):
    ref.child(user_id).update(updated_data)
    print(f'User {user_id} updated successfully.')

# Delete a record
def delete_user(user_id):
    ref.child(user_id).delete()
    print(f'User {user_id} deleted successfully.')



def read_user_by_credentials(email, password):
    users = ref.get()  # Get all users under the "registration" node
    if users:
        # Check if 'users' is a list or a dictionary
        if isinstance(users, list):
            for user in users:
                if user and user.get('email') == email and user.get('password') == password:
                    print(f'User found: {user}')
                    return user
        elif isinstance(users, dict):
            for user_id, user_data in users.items():
                if user_data['email'] == email and user_data['password'] == password:
                    print(f'User found: {user_data}')
                    return user_data
    print('No user found with the provided email and password.')
    return None

# # # Create
# create_user('2', 'aswin', '1234567890', 'john@example.com', 'password123')
# print("success")
# # # Read
# # read_user('1')

# # # Update
# # update_user('1', {'name': 'Johnathan Doe', 'email': 'johnathan@example.com'})

# # # Read again
# # read_user('1')

# # # Delete
# # delete_user('1')

# read_user_by_credentials('john@example.com', "password123")
