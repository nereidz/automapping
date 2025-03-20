import streamlit as st
import psycopg2
from rag import map_source_to_target  


def get_db_connection():
    try:
        return psycopg2.connect(
            host="localhost",
            database="test",
            user="postgres",
            password="sid"
        )
    except psycopg2.Error as e:
        st.error(f"Database connection error: {e}")
        return None


def get_categories():
    conn = get_db_connection()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT clientid FROM test")
        categories = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return categories
    except psycopg2.Error as e:
        st.error(f"Error fetching categories: {e}")
        return []

def get_items(category):
    conn = get_db_connection()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT layoutid FROM test WHERE clientid = %s", (category,))
        items = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return items
    except psycopg2.Error as e:
        st.error(f"Error fetching items: {e}")
        return []


st.title("AUTO MAPPING")

categories = get_categories()
selected_category = st.selectbox("Select ClientID", categories) if categories else None

if selected_category:
    items = get_items(selected_category)
    selected_item = st.selectbox("Select LayoutID", items) if items else None

if st.button("Generate Mapping"):
    if selected_category and selected_item:
        result= map_source_to_target(selected_category, selected_item) 
        st.write(result)  
       
    else:
        st.warning("Please select both ClientID and LayoutID.")
