"""This file conains the utility functions and classes for managing databases."""

# TODO(pourreza): spport other database types by changing sqlite3 to sqlalchemy

import logging
from typing import Any, Dict, List, Optional

from sqlglot import exp
from sqlglot import parse_one
from sqlglot.errors import ParseError
from sqlglot.optimizer.qualify import qualify

from .context_manager_utils import load_tables_description
from .database_schema_generator import DatabaseSchemaGenerator
from .db_info import get_db_all_tables
from .db_info import get_db_schema
from .db_info import get_table_all_columns
from .schema import DatabaseSchema


def schema_linking_scorer(
    gold_query: str,
    predicted_query: str,
):
  gt_schema_set = set()
  pred_schema_set = set()
  try:
    for column in parse_one(gold_query, read='sqlite').find_all(exp.Column):
      gt_schema_set.add(column.alias_or_name)
    for table in parse_one(gold_query, read='sqlite').find_all(exp.Table):
      gt_schema_set.add(table.name)
    for column in parse_one(predicted_query, read='sqlite').find_all(exp.Column):
      pred_schema_set.add(column.alias_or_name)
    for table in parse_one(predicted_query, read='sqlite').find_all(exp.Table):
      pred_schema_set.add(table.name)
  except Exception as e:
    # logging.error(f"Error in schema linking scorer: {e}")
    return 0.0
  return len(gt_schema_set.intersection(pred_schema_set)) / len(gt_schema_set)


def get_database_schema_string(
    db_path: str,
    db_id: str,
    tentative_schema: Dict[str, List[str]],
    schema_with_examples: Dict[str, List[str]],
    schema_with_descriptions: Dict[str, Dict[str, Dict[str, Any]]],
    include_value_description: bool,
) -> str:
  """Generates a schema string for the database.

  Args:
      db_path (str): The path to the database file.
      db_id (str): The database identifier.
      tentative_schema (Dict[str, List[str]]): The tentative schema.
      schema_with_examples (Dict[str, List[str]]): Schema with example values.
      schema_with_descriptions (Dict[str, Dict[str, Dict[str, Any]]]): Schema
        with descriptions.
      include_value_description (bool): Whether to include value descriptions.

  Returns:
      str: The generated schema string.
  """
  schema_generator = DatabaseSchemaGenerator(
      tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
      schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(
          schema_with_examples
      )
      if schema_with_examples
      else None,
      schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(
          schema_with_descriptions
      )
      if schema_with_descriptions
      else None,
      db_id=db_id,
      db_path=db_path,
  )
  schema_string = schema_generator.generate_schema_string(
      include_value_description=include_value_description
  )
  return schema_string


def get_sql_tables(db_path: str, sql: str) -> List[str]:
  """Retrieves table names involved in an SQL query.

  Args:
      db_path (str): Path to the database file.
      sql (str): The SQL query string.

  Returns:
      List[str]: List of table names involved in the SQL query.
  """
  db_tables = get_db_all_tables(db_path)
  try:
    parsed_tables = list(parse_one(sql, read='sqlite').find_all(exp.Table))
    correct_tables = [
        str(table.name).strip().replace('"', '').replace('`', '')
        for table in parsed_tables
        if str(table.name).strip().lower()
        in [db_table.lower() for db_table in db_tables]
    ]
    return correct_tables
  except Exception as e:
    logging.critical('Error in get_sql_tables: %s\nSQL: %s', e, sql)
    raise e


def _get_main_parent(expression: exp.Expression) -> Optional[exp.Expression]:
  """Retrieves the main parent expression for a given SQL expression.

  Args:
      expression (exp.Expression): The SQL expression.

  Returns:
      Optional[exp.Expression]: The main parent expression or None if not found.
  """
  parent = expression.parent
  while parent and not isinstance(parent, exp.Subquery):
    parent = parent.parent
  return parent


def _get_table_with_alias(
    parsed_sql: exp.Expression, alias: str
) -> Optional[exp.Table]:
  """Retrieves the table associated with a given alias.

  Args:
      parsed_sql (exp.Expression): The parsed SQL expression.
      alias (str): The table alias.

  Returns:
      Optional[exp.Table]: The table associated with the alias or None if not
      found.
  """
  return next(
      (
          table
          for table in parsed_sql.find_all(exp.Table)
          if table.alias == alias
      ),
      None,
  )


def get_sql_columns_dict(db_path: str, sql: str) -> Dict[str, List[str]]:
  """Retrieves a dictionary of tables and their respective columns involved in an SQL query.

  Args:
      db_path (str): Path to the database file.
      sql (str): The SQL query string.

  Returns:
      Dict[str, List[str]]: Dictionary of tables and their columns.
  """
  sql = (
      qualify(
          parse_one(sql, read='sqlite'),
          qualify_columns=True,
          validate_qualify_columns=False,
      )
      if isinstance(sql, str)
      else sql
  )
  columns_dict = {}

  sub_queries = [subq for subq in sql.find_all(exp.Subquery) if subq != sql]
  for sub_query in sub_queries:
    subq_columns_dict = get_sql_columns_dict(db_path, sub_query)
    for table, columns in subq_columns_dict.items():
      if table not in columns_dict:
        columns_dict[table] = columns
      else:
        columns_dict[table].extend([
            col
            for col in columns
            if col.lower() not in [c.lower() for c in columns_dict[table]]
        ])

  for column in sql.find_all(exp.Column):
    column_name = column.name
    table_alias = column.table
    table = _get_table_with_alias(sql, table_alias) if table_alias else None
    table_name = table.name if table else None

    if not table_name:
      candidate_tables = [
          t
          for t in sql.find_all(exp.Table)
          if _get_main_parent(t) == _get_main_parent(column)
      ]
      for candidate_table in candidate_tables:
        table_columns = get_table_all_columns(db_path, candidate_table.name)
        if column_name.lower() in [col.lower() for col in table_columns]:
          table_name = candidate_table.name
          break
    if table_name:
      if table_name not in columns_dict:
        columns_dict[table_name] = []
      if column_name.lower() not in [
          c.lower() for c in columns_dict[table_name]
      ]:
        columns_dict[table_name].append(column_name)

  return columns_dict


def get_union_schema_dict(
    schema_dict_list: List[Dict[str, List[str]]],
    db_path: str,
) -> Dict[str, List[str]]:
  """Unions a list of schemas.

  Args:
      schema_dict_list (List[Dict[str, List[str]]): The list of schemas.
      db_path (str): Path to the database file.

  Returns:
      Dict[str, List[str]]: The unioned schema.
  """
  full_schema = DatabaseSchema.from_schema_dict(get_db_schema(db_path))
  actual_name_schemas = []
  for schema in schema_dict_list:
    subselect_schema = full_schema.subselect_schema(
        DatabaseSchema.from_schema_dict(schema)
    )
    schema_dict = subselect_schema.to_dict()
    actual_name_schemas.append(schema_dict)
  union_schema = {}
  for schema in actual_name_schemas:
    for table, columns in schema.items():
      if table not in union_schema:
        union_schema[table] = columns
      else:
        union_schema[table] = list(set(union_schema[table] + columns))
  return union_schema


def get_database_schema(
    db_id: str,
    bird_database_path: str,
    schema_with_examples: Dict[str, List[str]],
    schema_with_descriptions: Dict[str, Dict[str, Dict[str, Any]]],
    queries: List[str] = None,
    include_value_description: bool = True,
    tentative_schema = None
) -> str:
  """Generates a schema string for the database.

  Args:
      db_id (str): The database identifier.
      bird_database_path (str): The path to the database file.
      schema_with_examples (Dict[str, List[str]]): Schema with example values.
      schema_with_descriptions (Dict[str, Dict[str, Dict[str, Any]]]): Schema
        with descriptions.
      queries (List[str]): The list of queries to generate the schema for.
      include_value_description (bool): Whether to include value descriptions.

  Returns:
      str: The generated schema string.
  """
  db_path = bird_database_path + '/' + db_id + '/' + db_id + '.sqlite'
  if queries:
    schema_dict_list = []
    for query in queries:
      try:
        schema_dict_list.append(get_sql_columns_dict(db_path, query))
      except (ParseError, KeyError, ValueError, TimeoutError) as e:
        schema_dict_list.append({})
    union_schema_dict = get_union_schema_dict(schema_dict_list, db_path)
  else:
    union_schema_dict = get_db_schema(db_path)
  if tentative_schema:
    union_schema_dict = get_union_schema_dict([tentative_schema], db_path)
  database_info = get_database_schema_string(
      db_path,
      db_id,
      union_schema_dict,
      schema_with_examples,
      schema_with_descriptions,
      include_value_description=include_value_description,
  )
  return database_info


def get_db_schema_db_id(db_id: str, bird_database_path: str, queries: List[str] = None, tentative_schema = None) -> str:
  """Generates a schema string for the database.

  Args:
      db_id (str): The database identifier.
      bird_database_path (str): The path to the database file.

  Returns:
      str: The generated schema string.
  """
  schema_with_descriptions = load_tables_description(bird_database_path, db_id)
  db_schema = get_database_schema(
      db_id,
      bird_database_path,
      schema_with_examples={},
      schema_with_descriptions=schema_with_descriptions,
      queries=queries,
      tentative_schema=tentative_schema,
  )
  return db_schema