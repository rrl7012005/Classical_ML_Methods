import re
import string
import json

word_distributions = {}
n = 2

pattern = re.compile(r'([{}])'.format(re.escape(string.punctuation)))


#Remove punctuations or treat as separate tokens. Remove capital letters.

def execute_file(filename, word_distributions, n):

    with open(filename, 'r', encoding='utf-8') as file:
        words = []
        for line in file:
            line = pattern.sub(r' \1 ', line)
            line = line.lower()
            for word in line.split():
                words.append(word)
                if len(words) == n + 1:
                    next_word = words[-1]
                    ngram = tuple(words[:-1])

                    # Update the word_distributions dictionary
                    if ngram in word_distributions:
                        if next_word in word_distributions[ngram]:
                            word_distributions[ngram][next_word] += 1
                        else:
                            word_distributions[ngram][next_word] = 1
                    else:
                        word_distributions[ngram] = {next_word: 1}

                    words.pop(0)

    return word_distributions

files = ['story.txt', 'story2.txt', 's3.txt', 's4.txt', 's5.txt', 's6.txt', 's7.txt', 's8.txt', 's9.txt', 's10.txt']

for x, filename in enumerate(files):
    print("ON FILE NUM {}".format(x))
    word_distributions = execute_file(filename, word_distributions, n)

#Compute probabilities
x = 0
for ngram, next_words in word_distributions.items():
    total_count = sum(next_words.values())
    x += 1
    print(x)
    for next_word, count in next_words.items():
        word_distributions[ngram][next_word] = count / total_count

string_key_distributions = {str(ngram): next_words for ngram, next_words in word_distributions.items()}

output_filename = 'word_distributions.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(string_key_distributions, f, ensure_ascii=False, indent=4)