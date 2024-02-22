-- Create a database if it doesn't exist
CREATE DATABASE kunalkp;

-- Switch to the database
USE kunalkp;

-- Create a table to store user information
CREATE TABLE users11 (
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL
);

-- Insert data into the table when a user signs up
-- Hash the password securely before inserting it (bcrypt or other secure hash function)
INSERT INTO users (username, email, password_hash)
VALUES (
    'user_input_username',
    'user_input_email',
    'hashed_user_input_password'
);
use kunalkp;
select * from users11;