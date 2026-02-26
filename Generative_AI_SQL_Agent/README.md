## Text-to-SQL AI Agent
This project is a Python-based utility that allows you to query a MySQL database using natural language. It leverages a local Gemma 3:1B model via Ollama to translate English questions into executable SQL queries, handling complex multi-table relationships automatically.

# Key Features
Local LLM Integration: Uses the ollama library to run inference locally, ensuring data privacy and eliminating API costs.

Dynamic Schema Discovery: Automatically fetches CREATE TABLE statements and foreign key constraints to provide the AI with structural context for accurate JOIN operations.

Automated Query Execution: A complete pipeline that accepts a user question, generates the SQL, cleans the output using regex, and retrieves results from the database.

# Project Structure
agent.py: The core Python script managing the LLM prompt engineering, database connection, and command-line loop.

db.sql: A sample schema including customers, products, and orders tables with pre-loaded data for testing.

# Setup & Usage
Install Dependencies:

Bash
pip install ollama mysql-connector-python
Prepare Database: Execute the db.sql script in your MySQL environment to create the store_db database.

Configure Credentials: Update the DB_CONFIG dictionary in agent.py with your MySQL username and password.

Run the Agent:

Bash
python agent.py
Example Query:

Input: "Who bought a Laptop?"

System: Generates a JOIN between customers and orders, executes it, and returns the result.