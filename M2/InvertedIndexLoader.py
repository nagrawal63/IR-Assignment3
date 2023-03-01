from InvertedIndex import InvertedIndex, Postings
import json

def loadInvertedIndexFromFile(filePath):
    data = {}
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            line_data = json.loads(line.rstrip('\n|\r'))
            token = list(line_data.keys())[0]
            data[token] = [Postings.from_json(value) for value in line_data[token]]
    return data