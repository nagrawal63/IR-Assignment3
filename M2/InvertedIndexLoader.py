import json

def loadInvertedIndexFromFile(filePath):
    data = {}
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            line_data = json.loads(line.rstrip('\n|\r'))
            token = list(line_data.keys())[0]
            from InvertedIndex import Postings
            data[token] = [Postings.from_json(value) for value in line_data[token]]
    return data

'''
Loads the inverted index from file(where inverted index is stored as jsonlines)
into a dictionary
'''
def loadInvertedIndexLineByLine(filePath):
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            data = {}
            line_data = json.loads(line.rstrip('\n|\r'))
            token = list(line_data.keys())[0]
            from InvertedIndex import Postings
            data[token] = [Postings.from_json(value) for value in line_data[token]]
            yield data

def getIndexEntryTrigram(token):
    fileName = "trigram_splitted_index/" + token[0] + ".json"
    filePtr = loadInvertedIndexLineByLine(fileName); line = next(filePtr)
    isFileEmpty = False

    while not isFileEmpty:
        try:
            if token == list(line.keys())[0]:
                return line
            line = next(filePtr)
        except StopIteration:
            isFileEmpty = True
    return None

def getIndexEntryBigram(token):
    fileName = "bigram_splitted_index/" + token[0] + ".json"
    filePtr = loadInvertedIndexLineByLine(fileName); line = next(filePtr)
    isFileEmpty = False

    while not isFileEmpty:
        try:
            if token == list(line.keys())[0]:
                return line
            line = next(filePtr)
        except StopIteration:
            isFileEmpty = True
    return None

def getIndexEntry(token):
    fileName = "splitted_index/" + token[0] + ".json"
    filePtr = loadInvertedIndexLineByLine(fileName); line = next(filePtr)
    isFileEmpty = False

    while not isFileEmpty:
        try:
            if token == list(line.keys())[0]:
                return line
            line = next(filePtr)
        except StopIteration:
            isFileEmpty = True
    return None

def getIndexDataAllTokens(tokens):
    from InvertedIndex import Postings
    stokens = sorted(tokens) # sorted for reducing loading time for same character
    prevc= None
    data_dict = {}
    for t in stokens:
        currentc = t[0]
        if currentc != prevc:
            with open(f'./splitted_index/{currentc}.json') as f:
                l = f.readlines()
                skippointer  = json.loads(l[-1]) # read skippointer first 
        prevc = currentc
        if(t not in skippointer.keys()):
            continue
        data = [Postings.from_json(d) for d in json.loads(l[skippointer[t]])[t]]  # read only part of file where token is 
        data_dict[t] = data 
    # return [data_dict[t] for t in tokens] # for keeping the original order
    return list(data_dict.values())

def getIndexDataAllTokensBigram(tokens):
    from InvertedIndex import Postings
    stokens = sorted(tokens) # sorted for reducing loading time for same character
    prevc= None
    data_dict = {}
    for t in stokens:
        currentc = t[0]
        if currentc != prevc:
            with open(f'./bigram_splitted_index/{currentc}.json') as f:
                l = f.readlines()
                skippointer  = json.loads(l[-1]) # read skippointer first
        prevc = currentc
        if(t not in skippointer.keys()):
            continue
        data = [Postings.from_json(d) for d in json.loads(l[skippointer[t]])[t]]  # read only part of file where token is
        data_dict[t] = data
    # return [data_dict[t] for t in tokens] # for keeping the original order
    return list(data_dict.values())

def getIndexDataAllTokensTrigram(tokens):
    from InvertedIndex import Postings
    stokens = sorted(tokens) # sorted for reducing loading time for same character
    prevc= None
    data_dict = {}
    for t in stokens:
        currentc = t[0]
        if currentc != prevc:
            with open(f'./trigram_splitted_index/{currentc}.json') as f:
                l = f.readlines()
                skippointer  = json.loads(l[-1]) # read skippointer first
        prevc = currentc
        if(t not in skippointer.keys()):
            continue
        data = [Postings.from_json(d) for d in json.loads(l[skippointer[t]])[t]]  # read only part of file where token is
        data_dict[t] = data
    # return [data_dict[t] for t in tokens] # for keeping the original order
    return list(data_dict.values())