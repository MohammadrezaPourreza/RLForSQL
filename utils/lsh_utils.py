"""This file contains the utility functions for LSH."""

import difflib
import logging
import os
import pathlib
import pickle
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import datasketch
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAIEmbeddings
import numpy as np
import tqdm

from .execution import execute_sql


aiplatform.init(
    project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_REGION")
)


EDIT_DISTANCE_SIMILARITY_THRESHOLD = 0.3

EMBEDDING_FUNCTION = VertexAIEmbeddings(model_name="text-embedding-004")
EMBEDDING_SIMILARITY_THRESHOLD = 0.6

LSH_CACHE_OBJECT = {}
MINHASHES_CACHE_OBJECT = {}


def _get_unique_values(db_path: str) -> Dict[str, Dict[str, List[str]]]:
  """Retrieves unique text values from the database excluding primary keys.

  Args:
      db_path (str): The path to the SQLite database file.

  Returns:
      Dict[str, Dict[str, List[str]]]: A dictionary containing unique values for
      each table and column.
  """
  table_names = [
      table[0]
      for table in execute_sql(
          db_path,
          "SELECT name FROM sqlite_master WHERE type='table';",
          fetch="all",
      )
  ]
  primary_keys = []

  for table_name in table_names:
    columns = execute_sql(
        db_path, f"PRAGMA table_info('{table_name}')", fetch="all"
    )
    for column in columns:
      if column[5] > 0:  # Check if it's a primary key
        column_name = column[1]
        if column_name.lower() not in [c.lower() for c in primary_keys]:
          primary_keys.append(column_name)

  unique_values: Dict[str, Dict[str, List[str]]] = {}
  for table_name in table_names:
    if table_name == "sqlite_sequence":
      continue
    logging.info("Processing %s", table_name)
    columns = [
        col[1]
        for col in execute_sql(
            db_path, f"PRAGMA table_info('{table_name}')", fetch="all"
        )
        if (
            "TEXT" in col[2]
            and col[1].lower() not in [c.lower() for c in primary_keys]
        )
    ]
    table_values: Dict[str, List[str]] = {}

    for column in columns:
      if any(
          keyword in column.lower()
          for keyword in [
              "_id",
              " id",
              "url",
              "email",
              "web",
              "time",
              "phone",
              "date",
              "address",
          ]
      ) or column.endswith("Id"):
        continue

      # TODO(b/290091111): Pre-cache unique values for each database

      try:
        result = execute_sql(
            db_path,
            f"""
                  SELECT SUM(LENGTH(unique_values)), COUNT(unique_values)
                  FROM (
                      SELECT DISTINCT `{column}` AS unique_values
                      FROM `{table_name}`
                      WHERE `{column}` IS NOT NULL
                  ) AS subquery
              """,
            fetch="one",
            timeout=480,
        )
      except sqlite3.Error as e:
        logging.exception("Error getting unique values for %s: %s", column, e)
        result = 0, 0

      sum_of_lengths, count_distinct = result
      if sum_of_lengths is None or count_distinct == 0:
        continue

      average_length = sum_of_lengths / count_distinct
      logging.info(
          "Column: %s, sum_of_lengths: %s, count_distinct: %s, average_length:"
          " %s",
          column,
          sum_of_lengths,
          count_distinct,
          average_length,
      )

      if (
          ("name" in column.lower() and sum_of_lengths < 5000000)
          or (sum_of_lengths < 2000000 and average_length < 25)
          or count_distinct < 100
      ):
        logging.info("Fetching distinct values for %s", column)
        try:
          values = [
              str(value[0])
              for value in execute_sql(
                  db_path,
                  f"SELECT DISTINCT `{column}` FROM `{table_name}` WHERE"
                  f" `{column}` IS NOT NULL",
                  fetch="all",
                  timeout=480,
              )
          ]
        except sqlite3.Error as e:
          logging.exception(
              "Error getting distinct values for %s: %s", column, e
          )
          values = []
        logging.info("Number of different values: %s", len(values))
        table_values[column] = values

    unique_values[table_name] = table_values

  return unique_values


def _create_minhash(
    signature_size: int, string: str, n_gram: int
) -> datasketch.MinHash:
  """Creates a MinHash object for a given string.

  Args:
      signature_size (int): The size of the MinHash signature.
      string (str): The input string to create the MinHash for.
      n_gram (int): The n-gram size for the MinHash.

  Returns:
      MinHash: The MinHash object for the input string.
  """
  m = datasketch.MinHash(num_perm=signature_size)
  for d in [string[i : i + n_gram] for i in range(len(string) - n_gram + 1)]:
    m.update(d.encode("utf8"))
  return m


def skip_column(column_name: str, column_values: List[str]) -> bool:
  """Determines whether to skip processing a column based on its values.

  Args:
      column_name (str): The name of the column.
      column_values (List[str]): The list of values in the column.

  Returns:
      bool: True if the column should be skipped, False otherwise.
  """
  if "name" in column_name.lower():
    return False
  sum_of_lengths = sum(len(value) for value in column_values)
  average_length = sum_of_lengths / len(column_values)
  return (sum_of_lengths > 50000) and (average_length > 20)


def make_lsh(
    unique_values: Dict[str, Dict[str, List[str]]],
    signature_size: int,
    n_gram: int,
    threshold: float,
) -> Tuple[
    datasketch.MinHashLSH, Dict[str, Tuple[datasketch.MinHash, str, str, str]]
]:
  """Creates a MinHash LSH from unique values.

  Args:
      unique_values (Dict[str, Dict[str, List[str]]]): The dictionary of unique
        values.
      signature_size (int): The size of the MinHash signature.
      n_gram (int): The n-gram size for the MinHash.
      threshold (float): The threshold for the MinHash LSH.

  Returns:
      Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The MinHash
      LSH object and the dictionary of MinHashes.
  """
  lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=signature_size)
  minhashes: Dict[str, Tuple[datasketch.MinHash, str, str, str]] = {}
  try:
    total_unique_values = 0
    for table_values in unique_values.values():
      for column_values in table_values.values():
        total_unique_values += len(column_values)
    logging.info("Total unique values: %s", total_unique_values)

    progress_bar = tqdm.tqdm(total=total_unique_values, desc="Creating LSH")

    for table_name, table_values in unique_values.items():
      for column_name, column_values in table_values.items():
        logging.info(
            "Processing %s - %s - %s",
            table_name,
            column_name,
            len(column_values),
        )

        for index, value in enumerate(column_values):
          minhash = _create_minhash(signature_size, value, n_gram)
          minhash_key = f"{table_name}_{column_name}_{index}"
          minhashes[minhash_key] = (minhash, table_name, column_name, value)
          lsh.insert(minhash_key, minhash)
          progress_bar.update(1)

    progress_bar.close()
  except (sqlite3.Error, ValueError) as e:
    logging.exception("Error creating LSH: %s", e)

  return lsh, minhashes


def make_db_lsh(db_directory_path: str, **kwargs: Any) -> None:
  """Creates a MinHash LSH for the database and saves the results.

  Args:
      db_directory_path (str): The path to the database directory.
      **kwargs (Any): Additional arguments for the LSH creation.
  """
  db_id = pathlib.Path(db_directory_path).name
  preprocessed_path = pathlib.Path(db_directory_path) / "preprocessed"
  preprocessed_path.mkdir(exist_ok=True)

  unique_values = _get_unique_values(
      str(pathlib.Path(db_directory_path) / f"{db_id}.sqlite")
  )
  logging.info("Unique values obtained")

  with open(preprocessed_path / f"{db_id}_unique_values.pkl", "wb") as file:
    pickle.dump(unique_values, file)
  logging.info("Saved unique values")

  lsh, minhashes = make_lsh(unique_values, **kwargs)

  with open(preprocessed_path / f"{db_id}_lsh.pkl", "wb") as file:
    pickle.dump(lsh, file)
  with open(preprocessed_path / f"{db_id}_minhashes.pkl", "wb") as file:
    pickle.dump(minhashes, file)


def set_lsh(db_directory_path: str, db_id: str) -> str:
  """Sets the LSH and minhashes attributes by loading from pickle files."""
  global LSH_CACHE_OBJECT, MINHASHES_CACHE_OBJECT
  if db_id not in LSH_CACHE_OBJECT:
    try:
      start_time = time.time()
      with (db_directory_path / "preprocessed" / f"{db_id}_lsh.pkl").open(
          "rb"
      ) as file:
        LSH_CACHE_OBJECT[db_id] = pickle.load(file)
      after_lsh_time = time.time()
      with (db_directory_path / "preprocessed" / f"{db_id}_minhashes.pkl").open(
          "rb"
      ) as file:
        MINHASHES_CACHE_OBJECT[db_id] = pickle.load(file)
      print(f"Time taken to load LSH {db_id}: {after_lsh_time - start_time}")
      print(
          f"Time taken to load Minhashes {db_id}:"
          f" {time.time() - after_lsh_time}"
      )
      return "success"
    except (pickle.PickleError, FileNotFoundError) as e:
      LSH_CACHE_OBJECT[db_id] = "error"
      MINHASHES_CACHE_OBJECT[db_id] = "error"
      print(f"Error loading LSH for {db_id}: {e}")
      return "error"
  elif LSH_CACHE_OBJECT[db_id] == "error":
    return "error"
  else:
    return "success"


def _jaccard_similarity(
    m1: datasketch.MinHash, m2: datasketch.MinHash
) -> float:
  """Computes the Jaccard similarity between two MinHash objects.

  Args:
      m1 (MinHash): The first MinHash object.
      m2 (MinHash): The second MinHash object.

  Returns:
      float: The Jaccard similarity between the two MinHash objects.
  """
  return m1.jaccard(m2)


def query_lsh(
    keyword: str,
    db_id: str,
    db_directory_path: str,
    signature_size: int = 100,
    n_gram: int = 3,
    top_n: int = 10,
) -> Dict[str, Dict[str, List[str]]]:
  """Queries the LSH for similar values to the given keyword.

  Keywords are extracted from the hint and the question using a single LLM call.

  Args:
      keyword (str): The keyword to search for.
      db_id (str): The ID of the database.
      db_directory_path (str): The path to the database directory.
      signature_size (int, optional): The size of the MinHash signature.
        Defaults to 20.
      n_gram (int, optional): The n-gram size for the MinHash. Defaults to 3.
      top_n (int, optional): The number of top results to return. Defaults to
        10.

  Returns:
      Dict[str, Dict[str, List[str]]]: A dictionary of similar values.

  Raises:
      Exception: If there is an error loading the LSH.
  """
  lsh_status = set_lsh(db_directory_path, db_id)
  if lsh_status == "success":
    minhashes = MINHASHES_CACHE_OBJECT[db_id]
    lsh = LSH_CACHE_OBJECT[db_id]
    query_minhash = _create_minhash(signature_size, keyword, n_gram)
    results = lsh.query(query_minhash)
    similarities = [
        (result, _jaccard_similarity(query_minhash, minhashes[result][0]))
        for result in results
    ]
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[
        :top_n
    ]

    similar_values_trimmed: Dict[str, Dict[str, List[str]]] = {}
    for result, _ in similarities:
      table_name, column_name, value = minhashes[result][1:]
      if table_name not in similar_values_trimmed:
        similar_values_trimmed[table_name] = {}
      if column_name not in similar_values_trimmed[table_name]:
        similar_values_trimmed[table_name][column_name] = []
      similar_values_trimmed[table_name][column_name].append(value)

    return similar_values_trimmed
  else:
    raise ValueError(f"Error loading LSH for {db_id}")


def _column_value(string: str) -> Tuple[Optional[str], Optional[str]]:
  """Splits a string into column and value parts if it contains '='.

  Args:
      string (str): The string to split.

  Returns:
      Tuple[Optional[str], Optional[str]]: The column and value parts.
  """
  if "=" in string:
    left_equal = string.find("=")
    first_part = string[:left_equal].strip()
    second_part = (
        string[left_equal + 1 :].strip()
        if len(string) > left_equal + 1
        else None
    )
    return first_part, second_part
  return None, None


def _get_to_search_values(keywords: List[str]) -> List[str]:
  """Extracts values to search from the keywords.

  Args:
      keywords (List[str]): The list of keywords.

  Returns:
      List[str]: A list of values to search.
  """

  def get_substring_packet(keyword: str, substring: str) -> Dict[str, str]:
    return {"keyword": keyword, "substring": substring}

  to_search_values = []
  for keyword in keywords:
    keyword = keyword.strip()
    to_search_values.append(get_substring_packet(keyword, keyword))
    if " " in keyword:
      for i in range(len(keyword)):
        if keyword[i] == " ":
          # Split the keyword into two parts if it contains a space.
          # For example, if the keyword is "John Doe", we will split it into
          # "John" and "Doe". This will allow us to search for similar values
          # for both "John" and "Doe".
          first_part = keyword[:i]
          second_part = keyword[i + 1 :]
          to_search_values.append(get_substring_packet(keyword, first_part))
          to_search_values.append(get_substring_packet(keyword, second_part))
    _, hint_value = _column_value(keyword)
    if hint_value:
      to_search_values.append(get_substring_packet(keyword, hint_value))
  to_search_values.sort(
      key=lambda x: (x["keyword"], len(x["substring"]), x["substring"]),
      reverse=True,
  )
  return to_search_values


def _get_similar_entities_via_lsh(
    substring_packets: List[Dict[str, str]],
    db_directory_path: str,
    db_id: str,
) -> List[Dict[str, Any]]:
  """Retrieves similar entities via LSH for the given substring packets.

  Args:
      substring_packets (List[Dict[str, str]]): The list of substring packets.
      db_directory_path (str): The path to the database directory.
      db_id (str): The ID of the database.

  Returns:
      List[Dict[str, Any]]: The list of similar entities.
  """
  similar_entities_via_lsh = []
  for packet in substring_packets:
    keyword = packet["keyword"]
    substring = packet["substring"]
    unique_similar_values = query_lsh(
        keyword=substring,
        db_id=db_id,
        db_directory_path=db_directory_path,
        signature_size=100,
        top_n=10,
    )
    for table_name, column_values in unique_similar_values.items():
      for column_name, values in column_values.items():
        for value in values:
          similar_entities_via_lsh.append({
              "keyword": keyword,
              "substring": substring,
              "table_name": table_name,
              "column_name": column_name,
              "similar_value": value,
          })
  return similar_entities_via_lsh


def _get_similar_entities_via_edit_distance(
    similar_entities_via_lsh: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
  """Filters similar entities via edit distance similarity.

  Args:
    similar_entities_via_lsh:

  Returns:
  """
  similar_entities_via_edit_distance_similarity = []
  for entity_packet in similar_entities_via_lsh:
    edit_distance_similarity = difflib.SequenceMatcher(
        None,
        entity_packet["substring"].lower(),
        entity_packet["similar_value"].lower(),
    ).ratio()
    if edit_distance_similarity >= EDIT_DISTANCE_SIMILARITY_THRESHOLD:
      entity_packet["edit_distance_similarity"] = edit_distance_similarity
      similar_entities_via_edit_distance_similarity.append(entity_packet)
  return similar_entities_via_edit_distance_similarity


def _get_similar_entities_via_embedding(
    similar_entities_via_edit_distance: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
  """Filters similar entities via embedding similarity.

  Args:
    similar_entities_via_edit_distance:

  Returns:
  """
  similar_values_dict = {}
  to_embed_strings = []
  for entity_packet in similar_entities_via_edit_distance:
    keyword = entity_packet["keyword"]
    substring = entity_packet["substring"]
    similar_value = entity_packet["similar_value"]
    if keyword not in similar_values_dict:
      similar_values_dict[keyword] = {}
    if substring not in similar_values_dict[keyword]:
      similar_values_dict[keyword][substring] = []
      to_embed_strings.append(substring)
    similar_values_dict[keyword][substring].append(entity_packet)
    to_embed_strings.append(similar_value)
  all_embeddings = EMBEDDING_FUNCTION.embed_documents(to_embed_strings)
  similar_entities_via_embedding_similarity = []
  index = 0
  for _, substring_dict in similar_values_dict.items():
    for _, entity_packets in substring_dict.items():
      substring_embedding = all_embeddings[index]
      index += 1
      similar_values_embeddings = all_embeddings[
          index : index + len(entity_packets)
      ]
      index += len(entity_packets)
      similarities = np.dot(similar_values_embeddings, substring_embedding)
      for i, entity_packet in enumerate(entity_packets):
        if similarities[i] >= EMBEDDING_SIMILARITY_THRESHOLD:
          entity_packet["embedding_similarity"] = similarities[i]
          similar_entities_via_embedding_similarity.append(entity_packet)
  return similar_entities_via_embedding_similarity


def get_similar_entities(
    keywords: List[str], db_directory_path: str, db_id: str
) -> Dict[str, Dict[str, List[str]]]:
  """Retrieves similar entities from the database based on keywords.

  Args:
      keywords (List[str]): The list of keywords.
      db_directory_path (str): The path to the database directory.
      db_id (str): The ID of the database.

  Returns:
      Dict[str, Dict[str, List[str]]]: A dictionary mapping table and column
      names to similar entities.
  """
  to_seartch_values = _get_to_search_values(keywords)
  similar_entities_via_lsh = _get_similar_entities_via_lsh(
      to_seartch_values, db_directory_path, db_id
  )
  similar_entities_via_edit_distance = _get_similar_entities_via_edit_distance(
      similar_entities_via_lsh
  )
  similar_entities_via_embedding = _get_similar_entities_via_embedding(
      similar_entities_via_edit_distance
  )
  selected_values = {}
  for entity in similar_entities_via_embedding:
    table_name = entity["table_name"]
    column_name = entity["column_name"]
    if table_name not in selected_values:
      selected_values[table_name] = {}
    if column_name not in selected_values[table_name]:
      selected_values[table_name][column_name] = []
    selected_values[table_name][column_name].append(entity)
  for table_name, column_values in selected_values.items():
    for column_name, values in column_values.items():
      max_edit_distance_similarity = max(
          entity["edit_distance_similarity"] for entity in values
      )
      values = [
          entity
          for entity in values
          if entity["edit_distance_similarity"]
          >= 0.9 * max_edit_distance_similarity
      ]
      max_embedding_similarity = max(
          entity["embedding_similarity"] for entity in values
      )
      selected_values[table_name][column_name] = [
          entity["similar_value"]
          for entity in values
          if entity["embedding_similarity"] >= 0.9 * max_embedding_similarity
      ]

  return selected_values