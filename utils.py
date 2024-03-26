import pandas as pd
from sqlalchemy import create_engine, Integer, String, Float, Column, MetaData, Table

def load_sqlite(data_fp, hr_fn, db_uri):
    hr_df = pd.read_excel(
    data_fp + hr_fn,
    dtype={
        "Term Year": int,
        "Term Month": int,
        "Job Title": str,
        "Business Unit": str,
        "main_quit_reason_text": str,
        "main_quit_reason_text_sentiment": str,
        "nps": int,
    },
    ).drop(columns=["Unnamed: 0"])

    hr_df.columns = [
        "term_year",
        "term_month",
        "job_title",
        "business_unit",
        "gender",
        "main_quit_reason_text",
        "main_quit_reason_text_sentiment",
        "nps",
    ]

    # Define your data types here based on your DataFrame structure
    data_types = {
        "term_year": Integer,
        "term_month": Integer,
        "job_title": String(255),
        "business_unit": String(255),
        "gender": String(50),
        "main_quit_reason_text": String(3000),
        "main_quit_reason_text_sentiment": String(50),
        "nps": Integer,
    }

    engine = create_engine(db_uri)
    conn = engine.connect()

    # Use SQLAlchemy's MetaData() to define the table structure based on your datatypes
    # metadata = MetaData()
    # exit_surveys = Table(
    #     "exit_surveys",
    #     metadata,
    #     *(Column(column, data_types[column]) for column in hr_df.columns),
    # )

    # metadata.create_all(engine)  # Creates the table

    hr_df.to_sql("exit_surveys", conn, if_exists="replace", index=False, dtype=data_types)
    select_cols = ",".join(hr_df.columns)
    select_q = f"SELECT {select_cols} FROM exit_surveys"
    return pd.read_sql(select_q, conn)
    

def read_from_sqlite(db_uri):
    engine = create_engine(db_uri)
    conn = engine.connect()
    # metadata.create_all(engine)  # Creates the table

    select_q = f"""SELECT term_year,
                        term_month,
                        job_title,
                        business_unit,
                        gender,
                        main_quit_reason_text,
                        main_quit_reason_text_sentiment,
                        nps
                 FROM exit_surveys"""
    return pd.read_sql(select_q, conn)
    