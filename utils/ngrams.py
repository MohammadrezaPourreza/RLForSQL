import re
import sqlglot

def extract_identifiers(expression):
    """
    Extracts column names, table names, and alias names from the SQLGlot AST.
    These identifiers will later be filtered out from the token set.
    """
    identifiers = set()

    # Extract column identifiers (e.g., "u.name" -> "u" and "name")
    # for col in expression.find_all(sqlglot.expressions.Column):
    #     # Split on dot to capture both table and column parts.
    #     for token in col.sql(dialect="sqlite").split('.'):
    #         identifiers.add(token.lower())

    # Extract table identifiers and table aliases
    for table in expression.find_all(sqlglot.expressions.Table):
        # The table expression itself (its name, which might be qualified)
        # for token in table.sql(dialect="sqlite").split('.'):
        #     identifiers.add(token.lower())
        # Check if the table has an alias via its arguments.
        alias_token = table.args.get("alias")
        if alias_token:
            # alias_token is an Expression; if it has a `name` attribute, use it.
            name = getattr(alias_token, "name", None)
            if name:
                identifiers.add(name.lower())
            else:
                identifiers.add(str(alias_token).lower())

    # Extract column aliases from Alias expressions, if any.
    for alias_expr in expression.find_all(sqlglot.expressions.Alias):
        alias_token = alias_expr.args.get("alias")
        if alias_token:
            name = getattr(alias_token, "name", None)
            if name:
                identifiers.add(name.lower())
            else:
                identifiers.add(str(alias_token).lower())

    return identifiers

def get_bag_of_words(query: str, n: int = 1) -> set:
    """
    Parse a SQLite query using SQLGlot, remove column names, table names,
    and aliases, and then return a set of tokens or n-grams based on the remaining words.
    
    :param query: The SQL query to parse.
    :param n: The n-gram size. Use n=1 for unigrams (bag-of-words).
    :return: A set of tokens (if n==1) or n-grams.
    """
    try:
        # Parse and normalize the query using SQLGlot.
        expression = sqlglot.parse_one(query, read="sqlite")
        normalized_query = expression.sql(dialect="sqlite")
        
        # Extract all identifiers (columns, tables, aliases) to filter out.
        identifiers = extract_identifiers(expression)
    except Exception as e:
        # If parsing fails, fall back to the original query and no filtering.
        normalized_query = query
        identifiers = set()
    
    # Tokenize the normalized query (case-insensitive).
    tokens = re.findall(r'\w+', normalized_query.lower())
    
    # Remove tokens that are identified as columns, tables, or aliases.
    filtered_tokens = [token for token in tokens if token not in identifiers and token != "as"]
    
    # Create n-grams from the filtered tokens.
    if n <= 1:
        return set(filtered_tokens)
    else:
        ngrams = set()
        for i in range(len(filtered_tokens) - n + 1):
            gram = " ".join(filtered_tokens[i:i+n])
            ngrams.add(gram)
        return ngrams

def jaccard_similarity(predicted_query: str, gold_query: str, n: int = 1) -> float:
    """
    Compute the Jaccard distance between two SQL queries based on their
    n-gram bags after filtering out columns and table aliases.
    
    The Jaccard distance is defined as:
        1 - (|tokens_pred ∩ tokens_gold| / |tokens_pred ∪ tokens_gold|)
    
    :param predicted_query: The predicted SQL query.
    :param gold_query: The reference (gold) SQL query.
    :param n: The n-gram size to use for tokenization.
    :return: The Jaccard distance as a float.
    """
    tokens_pred = get_bag_of_words(predicted_query, n)
    tokens_gold = get_bag_of_words(gold_query, n)
    
    intersection = tokens_pred.intersection(tokens_gold)
    union = tokens_pred.union(tokens_gold)
    
    if not union:
        return 0.0
    
    similarity = len(intersection) / len(union)
    return similarity