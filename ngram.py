def tokenize(line):
    # Break sentence in the token, remove empty tokens
    tokens = []
    for token in line.split(" "):
        if token != "":
            tokens.append(token)
    return tokens


def generate_ngrams(lines, n):
    ngrams = []
    for line in lines:
        tokens = tokenize(line)
        # example:
        # tokens = ["one", "two", "three", "four", "five"]

        sequences = [tokens[i:] for i in range(n)]
        # n = 3
        # The above will generate sequences of tokens starting from different elements of the list of tokens.
        # The parameter in the range() function controls how many sequences to generate.
        #
        # sequences = [
        #   ['one', 'two', 'three', 'four', 'five'],
        #   ['two', 'three', 'four', 'five'],
        #   ['three', 'four', 'five']]

        ngrams_zip = zip(*sequences)
        # The zip function takes the sequences as a list of inputs
        # (using the * operator, this is equivalent to zip(sequences[0], sequences[1], sequences[2]).
        # Each tuple it returns will contain one element from each of the sequences.
        #
        # To inspect the content of bigrams, try:
        # print(list(bigrams))
        # which will give the following:
        #
        # [
        #   ('one', 'two', 'three'),
        #   ('two', 'three', 'four'),
        #   ('three', 'four', 'five')
        # ]

        ngrams.extend(list(ngrams_zip))
    return ngrams


def ngram_to_string(ngram):
    return " ".join(ngram)
