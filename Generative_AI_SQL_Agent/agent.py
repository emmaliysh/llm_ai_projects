import ollama
import mysql.connector
import re

# --- Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',          # Replace with your MySQL username
    'password': 'password123',  # Replace with your MySQL password
    'database': 'store_db',   # Switched to our new database
}
MODEL_NAME = 'gemma3:1b'

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_full_schema():
    """Dynamically fetches the CREATE TABLE statements for ALL tables in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Get all table names
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    
    schema_str = ""
    # 2. Get the schema for each table
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SHOW CREATE TABLE {table_name}")
        create_stmt = cursor.fetchone()[1]
        schema_str += create_stmt + ";\n\n"
        
    cursor.close()
    conn.close()
    return schema_str

def generate_sql(question, schema):
    """Uses Ollama to generate SQL, handling potential JOINs."""
    prompt = f"""
    You are an expert MySQL database assistant. 
    Your task is to convert the user's natural language question into a valid MySQL query.
    
    Here is the schema for the database. Pay close attention to the FOREIGN KEYs to understand how to JOIN tables:
    {schema}
    
    Rules:
    1. ONLY return the SQL query. 
    2. Do not include any explanations, markdown formatting, or introductory text. 
    3. Ensure the syntax is correct for MySQL.
    4. If a question requires data from multiple tables, use the appropriate JOINs.
    
    User Question: {question}
    SQL Query:
    """

    response = ollama.chat(model=MODEL_NAME, messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    raw_output = response['message']['content'].strip()
    
    # Strip markdown blocks
    clean_sql = re.sub(r"```sql\n?", "", raw_output)
    clean_sql = re.sub(r"```\n?", "", clean_sql).strip()
    
    return clean_sql

def execute_query(sql):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    except mysql.connector.Error as err:
        return f"Error executing query: {err}"

def main():
    print("Multi-Table Agent is ready! Type 'exit' to quit.")
    # Fetch the schema once at startup
    full_schema = get_full_schema() 
    
    while True:
        question = input("\nAsk a question about customers, products, or orders: ")
        if question.lower() == 'exit':
            break
            
        print(f"Thinking...")
        
        sql_query = generate_sql(question, full_schema)
        print(f"\n[Generated SQL]:\n{sql_query}")
        
        results = execute_query(sql_query)
        
        print("\n[Results]:")
        if isinstance(results, list):
            if len(results) == 0:
                print("No records found.")
            else:
                for row in results:
                    print(row)
        else:
            print(results) 

if __name__ == "__main__":
    main()