import os
import json
import psycopg2
import pandas as pd
import sqlalchemy as sq
from sqlalchemy import text


def load_config(section=None):
    """loading configuration information"""
    
    config_file = os.path.join("config.json")
    # for debugging
    # config_file = os.path.join("/workspaces/zfkd_analytics/config.json")
    
    with open(config_file, 'r', encoding='utf-8') as file:
        config = json.load(file)
    if section:
        return config.get(section, {})
    
    return config


def create_cohort(cohort):
    """
    Create specified cohort as table in OMOP database
    """
    # sql cohort path
    sql_file_path = os.path.join("src/sql/", f"{cohort}.sql")
    # debugging
    # sql_file_path = os.path.join("sql/", f"{cohort}.sql")
    
    connection_params = load_config('database')
    
    # db connection and execution
    with psycopg2.connect(**connection_params) as conn:
        with conn.cursor() as cursor:
            # read SQL-Data 
            with open(sql_file_path, 'r', encoding='utf-8') as file:
                sql = file.read()
                cursor.execute(sql)
                
        conn.commit()

def df_import(df, table_name):
    """
    Create a table in the PostgreSQL database from dataframe
    """
    connection_params = load_config('database')  
    
    # database connection with SQLAlchemy
    engine = sq.create_engine(f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}")
    
    # truncate table to avoid ddl error
    with engine.connect() as connection:
        result = connection.execute(
                                text("""
                                    SELECT EXISTS (
                                    SELECT 1 
                                    FROM information_schema.tables 
                                    WHERE table_name = :table_name
                                    AND table_schema = 'cdm' 
                                    ) AS table_exists;
                                    """), {"table_name": table_name}
                                    )
        exists = result.fetchone()[0]

        if exists:
            connection.execute(text(f'TRUNCATE TABLE cdm.{table_name} CASCADE;'))
            connection.commit()

        # save dataframe in database
        df.to_sql(table_name, con=engine, schema='cdm', index=False, if_exists='append')
        
def load_cohort(cohort):
    """
    Load specified cohort information form OMOP database to python environment
    """
    connection_params = load_config('database')
    
    with psycopg2.connect(**connection_params) as conn:
        df = pd.read_sql_query(f"SELECT * FROM cdm.analytics_{cohort};", conn)
    
    return df
def execute_sql(statement):
    
    try:
        # db connection
        connections_params = load_config('database')
        conn = psycopg2.connect(**connections_params)
        cur = conn.cursor()

        # execute statement
        cur.execute(statement)

        # check if the query returns data (i.e., SELECT or WITH ... SELECT)
        if cur.description:
            # extract column names
            colnames = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=colnames)
            cur.close()
            conn.close()
            return df
        else:
            # for INSERT, UPDATE, etc.
            conn.commit()
            cur.close()
            conn.close()
            return None

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Execution error: {e}")
        return None

def delete_table(table_name):
    """
    This function deletes a specified table from the database.
    """
    # db connection
    connections_params = load_config('database')
    
    try:
        with psycopg2.connect(**connections_params) as conn:
            with conn.cursor() as cur:
                sql = f"DROP TABLE IF EXISTS cdm.{table_name} CASCADE;"  # delete with dependencies
                cur.execute(sql)
                conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except:
            pass
