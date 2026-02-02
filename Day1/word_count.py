from collections import Counter

def word_counter(text):
    tokens = text.lower().split()
    return Counter(tokens)

if __name__ == "__main__":
    text = "I love NPL and I love machine learning."
    counters = word_counter(text)
    for word, freq in counters.items():
        print(word, freq)