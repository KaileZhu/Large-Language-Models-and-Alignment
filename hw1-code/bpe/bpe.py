import re
from collections import Counter

class Tokenizer:
    def __init__(self):
        self.bpe_vocab = {}  # 存储BPE词汇
        self.char_to_id = {}  # 字符到ID的映射
        self.id_to_char = {}  # ID到字符的映射
    
    def _get_stats(self, text):
        """ 获取文本中所有字符对的频率 """
        pairs = Counter()
        words = text.split()
        for word in words:
            symbols = list(word) + ['</w>']  # 词的结尾标记为</w>
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += 1
        return pairs
    
    def _merge_vocab(self, pair, vocab):
        """ 合并词汇对，并更新词汇表 """
        first, second = pair
        replacement = ''.join(pair)
        new_vocab = {}
        pattern = re.escape(' '.join(pair))
        replacement_pattern = replacement + ' '  # 新词用空格分开

        for word in vocab:
            new_word = re.sub(pattern, replacement_pattern, ' '.join(word))
            new_vocab[new_word] = vocab[word]
        
        return new_vocab
    
    def train(self, text, vocab_size):
        """ 训练BPE tokenizer """
        # 1. 准备输入数据
        text = text.lower()  # 统一转换为小写
        words = text.splitlines()
        vocab = {}
        
        # 2. 统计所有字符对
        for word in words:
            word = word + ' </w>'  # 词尾添加标志
            vocab[word] = vocab.get(word, 0) + 1
        
        # 3. 持续合并字符对直到达到词汇大小
        while len(vocab) < vocab_size:
            pairs = self._get_stats(text)  # 获取字符对的统计
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)  # 找到出现最频繁的字符对
            vocab = self._merge_vocab(best_pair, vocab)  # 合并字符对
            
        # 4. 更新BPE词汇表
        self.bpe_vocab = vocab
    
    def encode(self, text):
        """ 将文本编码为token id """
        tokens = []
        words = text.split()
        for word in words:
            tokens.append(self.bpe_vocab.get(word, word))
        return tokens

    def decode(self, ids):
        """ 将token id解码为文本 """
        decoded_text = ' '.join([self.id_to_char.get(i, '') for i in ids])
        return decoded_text




# 测试文本
text = """
low low lower lowest
new newt newts
"""

# 创建Tokenizer实例
tokenizer = Tokenizer()

# 训练BPE模型
vocab_size = 10
tokenizer.train(text, vocab_size)

# 输出训练后的BPE词汇表
print("BPE Vocabularies:")
for word, count in tokenizer.bpe_vocab.items():
    print(f"{word}: {count}")

# 编码测试
encoded = tokenizer.encode("low newt")
print("\nEncoded:", encoded)

# 解码测试
decoded = tokenizer.decode(encoded)
print("\nDecoded:", decoded)
