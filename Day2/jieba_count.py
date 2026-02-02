import jieba
from collections import Counter

text = """
我爱自然语言处理。
自然语言处理是人工智能的重要方向。
AIGC正在改变世界。
"""

words = jieba.lcut(text)
vocab = Counter(words)

print(words)
print(len(vocab))
print(vocab.most_common(10))