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

def getIndexEntry(token):
    fileName = "splitted_index/" + token[0] + ".json"
    data = {}
    with open(fileName, "r") as file:
        filelines = file.readlines()
        tokenMap = json.loads(filelines[-1])
        indexLine = filelines[tokenMap[token]-1]
        line_data = json.loads(indexLine.rstrip('\n|\r'))
        token = list(line_data.keys())[0]
        from InvertedIndex import Postings
        data[token] = [Postings.from_json(value) for value in line_data[token]]
    # print(data)
    return data