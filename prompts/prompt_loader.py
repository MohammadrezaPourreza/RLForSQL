"""This code contains the source code for the prompt loader utility."""

import logging
import os

TEMPLATES_ROOT_PATH = os.path.dirname(__file__)


def load_prompt(template_name: str) -> str:
  """Loads a template from a file.

  Args:
      template_name (str): The name of the template to load.

  Returns:
      str: The content of the template.
  """

  file_name = f"{template_name}.txt"
  template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)

  try:
    with open(template_path, "r") as file:
      template = file.read()
    logging.info("Template %s loaded successfully.", template_name)
    return template
  except FileNotFoundError:
    logging.exception("Template file not found: %s", template_path)
    raise
  except Exception as e:
    logging.exception("Error loading template %s: %s", template_name, e)
    raise