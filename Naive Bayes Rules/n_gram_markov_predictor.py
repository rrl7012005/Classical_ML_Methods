import json
import ast
import random

input_filename = 'word_distributions.json'
with open(input_filename, 'r', encoding='utf-8') as f:
    string_key_distributions = json.load(f)

word_distributions = {ast.literal_eval(ngram): next_words for ngram, next_words in string_key_distributions.items()}

input_string = input("Please write your string: ")
n = 25
N = 2

input_string = input_string.lower()
input_arr = input_string.split()

for i in range(n):
    input_tuple = tuple(input_arr[-N:])
    next_word_distribution = word_distributions[input_tuple]
    most_probable_word = max(next_word_distribution, key=next_word_distribution.get)
    probable_words = [word for word, count in next_word_distribution.items() if count == next_word_distribution[most_probable_word]]
    print(probable_words)
    next_word = random.choice(probable_words)
    input_arr.append(next_word)

output_string = " ".join(input_arr)
print(output_string)