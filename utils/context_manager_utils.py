"""This file contains the utility functions for context manager."""

import logging
import pathlib
from typing import Dict

import pandas as pd


def load_tables_description(
    db_directory_path: str, db_id: str, use_value_description: bool = True
) -> Dict[str, Dict[str, Dict[str, str]]]:
  """Loads table descriptions from CSV files in the database directory.

  Args:
      db_directory_path (str): The path to the database directory.
      db_id (str): The ID of the database.
      use_value_description (bool): Whether to include value descriptions.

  Returns:
      Dict[str, Dict[str, Dict[str, str]]]: A dictionary containing table
      descriptions.
  """
  encoding_types = ['utf-8-sig', 'cp1252']
  description_path = (
      pathlib.Path(db_directory_path + '/' + db_id) / 'database_description'
  )

  if not description_path.exists():
    logging.warning('Description path does not exist: %s', description_path)
    return {}

  table_description = {}
  for csv_file in description_path.glob('*.csv'):
    table_name = csv_file.stem.lower().strip()
    table_description[table_name] = {}
    could_read = False
    for encoding_type in encoding_types:
      try:
        table_description_df = pd.read_csv(
            csv_file, index_col=False, encoding=encoding_type
        )
        for _, row in table_description_df.iterrows():
          column_name = row['original_column_name']
          expanded_column_name = (
              row.get('column_name', '').strip()
              if pd.notna(row.get('column_name', ''))
              else ''
          )
          if pd.notna(row.get('column_description', '')):
            column_description = (
                row.get('column_description', '')
                .replace('\n', ' ')
                .replace('commonsense evidence:', '')
                .strip()
            )
          else:
            column_description = ''
          data_format = (
              row.get('data_format', '').strip()
              if pd.notna(row.get('data_format', ''))
              else ''
          )
          value_description = ''
          if use_value_description and pd.notna(
              row.get('value_description', '')
          ):
            value_description = (
                row['value_description']
                .replace('\n', ' ')
                .replace('commonsense evidence:', '')
                .strip()
            )
            if value_description.lower().startswith('not useful'):
              value_description = value_description[10:].strip()

          table_description[table_name][column_name.lower().strip()] = {
              'original_column_name': column_name,
              'column_name': expanded_column_name,
              'column_description': column_description,
              'data_format': data_format,
              'value_description': value_description,
          }
        logging.info(
            'Loaded descriptions from %s with encoding %s',
            csv_file,
            encoding_type,
        )
        could_read = True
        break
      except UnicodeDecodeError as e:
        # logging.exception('Error decoding %s: %s', csv_file, e)
        continue
      except pd.errors.EmptyDataError as e:
        # logging.exception('Empty data in %s: %s', csv_file, e)
        continue
      except pd.errors.ParserError as e:
        logging.exception('Error parsing %s: %s', csv_file, e)
        continue
    if not could_read:
      logging.warning('Could not read descriptions from %s', csv_file)
    return table_description