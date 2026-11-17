"""The prediction log (D-06): one row per prediction in PostgreSQL, with the model version."""
import logging
import os

import psycopg

logger = logging.getLogger(__name__)

INSERT_SQL = """
    INSERT INTO predictions (engine_id, cycle, rul, model_version)
    VALUES (%s, %s, %s, %s)
"""


def save_prediction(engine_id, cycle, rul, model_version):
    """Store one prediction. A missing or broken database is logged, never raised: logging must not stop serving."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.debug("DATABASE_URL not set; prediction not stored")
        return
    try:
        with psycopg.connect(url) as conn:
            conn.execute(INSERT_SQL, (engine_id, cycle, rul, model_version))
    except psycopg.Error:
        logger.exception("Could not store the prediction")
