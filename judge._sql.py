import pandas as pd
import logging
import psycopg2
import csv
from concurrent.futures import ThreadPoolExecutor
import time
import os

# Configure logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# Define the database connection parameters
connection_params = {
    "host": "id-hdb-psgr-cp7.ethz.ch",
    "dbname": "led",
    "user": "lixiang",
    'password': 'Wmhzgjwmhxgj99'  # Replace with your actual password
}

# Updated SQL query
query = """
SELECT opinion, opinion_type, citation_count, opinion_songernames,
       sentence_count, word_count, char_count, words_per_sentence, 
       words_per_paragraph, dc_identifier, date_part('year', date_standard) AS year, 
       court_normalized
FROM
    lexis_opinions_circuit
NATURAL JOIN lexis_cases_circuit
WHERE date_part('year', date_standard) BETWEEN 1970 AND 2009;
"""

# Output file paths for different decades
output_files = {
    "1970s": "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_1970s.csv",
    "1980s": "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_1980s.csv",
    "1990s": "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_1990s.csv",
    "2000s": "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_2000s.csv",
}

def get_decade(year):
    try:
        year = int(year)
        if 1970 <= year < 1980:
            return "1970s"
        elif 1980 <= year < 1990:
            return "1980s"
        elif 1990 <= year < 2000:
            return "1990s"
        elif 2000 <= year < 2010:
            return "2000s"
    except ValueError:
        logging.warning(f"Invalid year value: {year}")
    return None

# Connect to the database and save results to CSV
logging.info("Connecting to the database...")
try:
    with psycopg2.connect(**connection_params) as conn:
        # Create a cursor
        logging.info("Creating a cursor and executing the query...")
        with conn.cursor() as cur:
            cur.execute(query)

            # Fetch all rows and column names
            rows = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]

            logging.info(f"Retrieved {len(rows)} rows. Saving to CSV...")

            # Create a dictionary to store data for each decade
            decade_data = {decade: [] for decade in output_files.keys()}

            # Sort rows into decades
            for row in rows:
                year = row[column_names.index('year')]
                decade = get_decade(year)
                if decade:
                    decade_data[decade].append(row)

            # Write data to CSV files for each decade
            for decade, data in decade_data.items():
                output_file = output_files[decade]
                with open(output_file, mode="w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(column_names)
                    writer.writerows(data)
                logging.info(f"Data for {decade} successfully saved to {output_file}.")

except Exception as e:
    logging.error(f"An error occurred: {e}")

logging.info("Done.")
