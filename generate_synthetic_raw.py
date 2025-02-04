import random
import dill

DICTIONARY_PATH = "10kwords.txt"

NUM_DOCS = 3000
NUM_QUERIES = NUM_DOCS
QUERY_WORDS_RANGE = (3, 6)
DOC_WORDS_RANGE = (300, 300)

def load_dictionary(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]
    return words

def generate_random_document(word_list, min_len, max_len):
    doc_length = random.randint(min_len, max_len)
    chosen_words = random.choices(word_list, k = doc_length)
    return " ".join(chosen_words)

def main():
    dictionary_words = load_dictionary(DICTIONARY_PATH)
    if not dictionary_words:
        raise ValueError(f"No words loaded from {DICTIONARY_PATH}.")

    D = []
    for doc_id in range(NUM_DOCS):
        content = generate_random_document(dictionary_words,
                                               DOC_WORDS_RANGE[0],
                                               DOC_WORDS_RANGE[1])
        D.append(content)
    with open("synt_docs.pkl", "wb") as file:
        dill.dump(D, file)
    
    del D
    Q = []
    for q_id in range(NUM_QUERIES):
        content = generate_random_document(dictionary_words,
                                            QUERY_WORDS_RANGE[0],
                                            QUERY_WORDS_RANGE[1])
        Q.append(content)
    with open("synt_queries.pkl", "wb") as file:
        dill.dump(Q, file)


    print(f"Generated {NUM_DOCS} synthetic docs and {NUM_QUERIES} queries")

if __name__ == "__main__":
    main()
