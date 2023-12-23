import math

from ngram import ngram_to_string
# ==================================================== write ========================================================= #


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


# ==================================================== read ========================================================= #

def arpa_read_file(filename):

    language_model = []
    with open(filename, 'r', encoding='utf-8') as f:
        header = True
        ngram_probabilities = None

        for line in f:
            line = line.strip()
            if len(line)==0: #Skip empty lines
                continue
            if header:
                if line=="\\data\\":
                    header=False
                continue
            if line=="\\end\\":
                break

            if line.startswith("ngram"):
                ngram_count=int(line.split("=")[1])
                ngram_probabilities={'unique_ngram_count':ngram_count,'dict':{}}
                language_model.append(ngram_probabilities)
                continue

            if line.startswith("\\"):
                continue

            parts = line.split()
            probability = float(parts[0])
            if parts[-1].isdigit():
                ngram_tuple = tuple(parts[1:-1])
                backoff_weight = float(parts[-1]) if len(parts) > 2 else None
                ngram_probabilities = {
                    ngram_tuple: {'value': math.pow(10, probability), 'backoff_weight': math.pow(10, backoff_weight)}}
            else:
                ngram_tuple = tuple(parts[1:])
                ngram_probabilities = {ngram_tuple: {'value': math.pow(10, probability)}}
            language_model[len(ngram_tuple) - 1]["dict"].update(ngram_probabilities)

        return language_model






