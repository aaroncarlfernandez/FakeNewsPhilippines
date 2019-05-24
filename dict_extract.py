def read_dic(filepath):
  # category_mapping is a mapping from integer string to category name
  category_mapping = {}
  # category_names is equivalent to category_mapping.values() but retains original ordering
  category_names = []
  lexicon = {}
  # the mode is incremented by each '%' line in the file
  mode = 0
  for line in open(filepath):
    tsv = line.strip()
    if tsv:
      parts = tsv.split()
      if parts[0] == '%':
        mode += 1
      elif mode == 1:
        # definining categories
        category_names.append(parts[1])
        category_mapping[parts[0]] = parts[1]
      elif mode == 2:
        lexicon[parts[0]] = [category_mapping[category_id] for category_id in parts[1:]]
  return lexicon, category_names


def _build_trie(lexicon):
  trie = {}
  for pattern, category_names in lexicon.items():
    cursor = trie
    for char in pattern:
      if char == '*':
        cursor['*'] = category_names
        break
      if char not in cursor:
        cursor[char] = {}
      cursor = cursor[char]
    cursor['$'] = category_names
  return trie


def _search_trie(trie, token, token_i=0):
  if '*' in trie:
    return trie['*']
  elif '$' in trie and token_i == len(token):
    return trie['$']
  elif token_i < len(token):
    char = token[token_i]
    if char in trie:
      return _search_trie(trie[char], token, token_i + 1)
  return []


def load_token_parser(filepath):
  lexicon, category_names = read_dic(filepath)
  trie = _build_trie(lexicon)

  def parse_token(token):
    for category_name in _search_trie(trie, token):
      yield category_name

  return parse_token, category_names