import math

from ngram import ngram_to_string


def write_arpa_file(language_model, filename="out.lm"):
    with open(f"../out/{filename}", 'w', encoding='utf-8') as f:
        # header
        f.write("\\data\\\n")
        f.writelines(arpa_content_header(language_model))
        f.write("\n")

        # content
        for ngram_probabilities in language_model:
            f.writelines(arpa_ngram_definition(ngram_probabilities))
            f.write("\n")

        f.write("\\end\\\n")


def arpa_content_header(language_model):
    for i in range(len(language_model)):
        yield f"ngram {i + 1}={language_model[i]['unique_ngram_count']}\n"


def arpa_ngram_definition(probabilities):
    yield f"\\{probabilities['n']}-grams:\n"
    for ngram in probabilities['dict'].keys():
        probability = probabilities['dict'][ngram]
        ngram_string = ngram_to_string(ngram)

        if probability['value'] == 0:
            output_probability = -99
        else:
            output_probability = math.log10(probability['value'])

        line = f"{output_probability} {ngram_string}"
        if "backoff-weight" in probability:
            if probability['backoff-weight'] == 0:
                backoff_weight = -99
            else:
                backoff_weight = math.log10(probability['backoff-weight'])
            line += f" {backoff_weight}"

        yield f"{line}\n"
