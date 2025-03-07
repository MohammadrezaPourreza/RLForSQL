"""Utility functions for executing SQL queries."""

import logging
import random
import re
import sqlite3
import threading
from typing import Any, Dict, List, Union

from func_timeout import func_timeout
from func_timeout import FunctionTimedOut
import sqlvalidator


def format_sql_query(query: str, meta_time_out: int = 10):
  """Formats the SQL query.

  Args:
      query (str): The SQL query to format.
      meta_time_out (int): The timeout for the formatting.

  Returns:
      str: The formatted SQL query.
  """
  try:
    return func_timeout(meta_time_out, sqlvalidator.format_sql, args=(query))
  except FunctionTimedOut:
    print(f"Timeout in format_sql_query: {query}")
    return query
  except Exception:  # pylint: disable=broad-exception-caught
    return query


def syntax_check_sql(
    db_path: str,
    db_id: str,
    query: str) -> bool:
  """Checks the syntax of an SQL query."""
  db_path = f"{db_path}/{db_id}/{db_id}.sqlite"
  try:
    with sqlite3.connect(db_path, timeout=60) as conn:
          cursor = conn.cursor()
          cursor.execute(query)
          return True
  except sqlite3.Error:
    return False
  except Exception as e:
    return False
  
_thread_local = threading.local()

def get_db_connection(db_path: str):
    """Creates a separate SQLite connection per thread and per database path."""
    if not hasattr(_thread_local, "connections"):
        _thread_local.connections = {}
    if db_path not in _thread_local.connections:
        _thread_local.connections[db_path] = sqlite3.connect(db_path, timeout=60, check_same_thread=False)
    return _thread_local.connections[db_path]

def execute_sql(
    db_path: str, sql: str, fetch: Union[str, int] = 5000, timeout: int = 60
) -> Any:
  """Executes a SQL query on a database.

  Args:
      db_path (str): The path to the database.
      sql (str): The SQL query to execute.
      fetch (Union[str, int]): The fetch mode. Can be "all", "one", "random", or
        an integer.
      timeout (int): The timeout for the query in seconds.

  Returns:
      Any: The result of the query.

  Raises:
      TimeoutError: If the query execution exceeds the timeout.
      sqlite3.Error: If an error occurs during the query execution.
  """

  sql = _clean_sql(sql)

  class QueryThread(threading.Thread):
    """A thread class for executing SQL queries."""

    def __init__(self):
      threading.Thread.__init__(self)
      self.result = None
      self.exception = None

    def run(self):
      try:
        # with sqlite3.connect(db_path, timeout=60) as conn:
          conn = get_db_connection(db_path)
          cursor = conn.cursor()
          cursor.execute(sql)
          if fetch == "all":
            self.result = cursor.fetchall()
          elif fetch == "one":
            self.result = cursor.fetchone()
          elif fetch == "random":
            samples = cursor.fetchmany(10)
            self.result = random.choice(samples) if samples else []
          elif isinstance(fetch, int):
            self.result = cursor.fetchmany(fetch)
          else:
            raise ValueError(
                "Invalid fetch argument. Must be 'all', 'one', 'random',"
                " or an integer."
            )
      except sqlite3.Error as e:
        self.exception = e

  query_thread = QueryThread()
  query_thread.start()
  query_thread.join(timeout)
  if query_thread.is_alive():
    raise TimeoutError(
        f"SQL query execution exceeded the timeout of {timeout} seconds."
    )
  if query_thread.exception:
    # logging.info(
    #     "Error in execute_sql: %s",
    #     query_thread.exception,
    # )
    raise query_thread.exception
  return query_thread.result


def _clean_sql(sql: str) -> str:
  """Cleans the SQL query by removing unwanted characters and whitespace.

  Args:
      sql (str): The SQL query string.

  Returns:
      str: The cleaned SQL query string.
  """
  sql = sql.replace("\n", " ").replace('"', "'")
  sql = re.sub(r'\\+([\'"])', r"\1", sql)
  return sql


def _string_based_query_clustering(queries: List[str]) -> List[str]:
  """Clusters the queries based on their string representation.

  Args:
      queries: A list of SQL queries.

  Returns:
      A list of clustered queries.
  """
  clustered_queries = {}
  for query in queries:
    formatted_query = format_sql_query(_clean_sql(query))
    if formatted_query not in clustered_queries:
      clustered_queries[formatted_query] = query
  return list(clustered_queries.keys())


def _compare_sqls_outcomes(
    db_path: str, predicted_sql: str, ground_truth_sql: str
) -> int:
  """Compares the outcomes of two SQL queries to check for equivalence.

  Args:
      db_path (str): The path to the database file.
      predicted_sql (str): The predicted SQL query.
      ground_truth_sql (str): The ground truth SQL query.

  Returns:
      int: 1 if the outcomes are equivalent, 0 otherwise.

  Raises:
      Exception: If an error occurs during SQL execution.
  """
  try:
    predicted_res = execute_sql(db_path, predicted_sql)
    ground_truth_res = execute_sql(db_path, ground_truth_sql)
    if len(set(predicted_res)) == 0 or len(set(ground_truth_res)) == 0:
      return 0
    return int(set(predicted_res) == set(ground_truth_res))
  except sqlite3.Error as e:
    # logging.exception("Error comparing SQL outcomes: %s", e)
    raise e


def compare_sqls(
    db_directory_path: str,
    db_id: str,
    predicted_sql: str,
    ground_truth_sql: str,
    meta_time_out: int = 30,
) -> Dict[str, Union[int, str]]:
  """Compares predicted SQL with ground truth SQL within a timeout.

  Args:
      db_directory_path (str): The path to the database directory.
      db_id (str): The ID of the database.
      predicted_sql (str): The predicted SQL query.
      ground_truth_sql (str): The ground truth SQL query.
      meta_time_out (int): The timeout for the comparison.

  Returns:
      dict: A dictionary with the comparison result and any error message.
  """
  db_path = f"{db_directory_path}/{db_id}/{db_id}.sqlite"
  predicted_sql = _clean_sql(predicted_sql)
  try:
    res = func_timeout(
        meta_time_out,
        _compare_sqls_outcomes,
        args=(db_path, predicted_sql, ground_truth_sql),
    )
    error = "incorrect answer" if res == 0 else "--"
  except FunctionTimedOut:
    # logging.warning("Comparison timed out.")
    error = "timeout"
    res = 0
  except sqlite3.Error as e:
    # logging.exception("Error comparing SQL outcomes: %s", e)
    error = str(e)
    res = 0
  return {"exec_res": res, "exec_err": error}