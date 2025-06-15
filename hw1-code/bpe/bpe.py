import re
import os
from collections import defaultdict


class Tokenizer:
    def __init__(self):
        # 初始化词汇表和BPE合并规则
        self.vocab = {}
        self.bpe_codes = {}
    
    def train(self, text, vocab_size):
        """
        使用BPE算法训练tokenizer。
        参数:
            text (str): 用于训练BPE的文本数据。
            vocab_size (int): 最终词汇表的大小。
        """
        # 将文本按空格分割成单词
        words = text.split()
        
        # 初始化词汇表，将每个字符当作一个独立的token
        vocab = defaultdict(int)
        for word in words:
            # 对每个单词将字符分割并添加一个特殊符号 </w> 表示词尾
            word = " ".join(list(word)) + " </w>"
            vocab[word] += 1
        
        # 开始BPE训练过程
        while len(vocab) < vocab_size:
            # 统计最常见的字符对
            pair_freq = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_freq[pair] += freq
            
            # 找到最常见的字符对
            if not pair_freq:
                break
            most_frequent_pair = max(pair_freq, key=pair_freq.get)
            
            # 将最常见的字符对合并为新的符号
            new_symbol = "".join(most_frequent_pair)
            self.bpe_codes[most_frequent_pair] = new_symbol
            
            # 更新词汇表，进行字符对的替换
            new_vocab = {}
            for word, freq in vocab.items():
                symbols = word.split()
                new_word = []
                i = 0
                while i < len(symbols) - 1:
                    pair = (symbols[i], symbols[i + 1])
                    if pair == most_frequent_pair:
                        new_word.append(new_symbol)  # 合并字符对
                        i += 2
                    else:
                        new_word.append(symbols[i])  # 保持其他部分不变
                        i += 1
                if i < len(symbols):
                    new_word.append(symbols[i])  # 添加最后一个符号
                new_vocab[" ".join(new_word)] = freq
            vocab = new_vocab
        
        # 最终得到的词汇表
        self.vocab = vocab

    def encode(self, text):
        """
        将输入文本编码为token ID列表。
        参数:
            text (str): 需要编码的文本数据。
        返回:
            ids (list): 对应的token ID列表。
        """
        # 将输入文本分割为字符列表，并添加词尾符号 </w>
        word = " ".join(list(text)) + " </w>"
        
        # 使用BPE编码规则将文本转换为token序列
        symbols = word.split()
        while len(symbols) > a:
            # 找到最匹配的字符对
            pair = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self.bpe_codes:
                    symbols[i] = self.bpe_codes[pair]  # 使用合并后的符号替换字符对
                    del symbols[i + 1]  # 删除旧的字符对
                    break
        
        # 将最终的符号序列转换为token ID
        return [self.vocab.get(symbol, 0) for symbol in symbols]

    def decode(self, ids):
        """
        将token ID列表解码为原始文本。
        参数:
            ids (list): token ID列表。
        返回:
            text (str): 解码后的文本。
        """
        # 创建反向词汇表（将ID映射回符号）
        rev_vocab = {v: k for k, v in self.vocab.items()}
        
        # 根据ID生成符号列表
        symbols = [rev_vocab.get(i, "") for i in ids]
        
        # 拼接符号并返回字符串
        return "".join(symbols).replace(" </w>", "")  # 移除词尾符号 </w>
    


if __name__ == "__main__":
    ##################################
    #########  测试 BPE算法  #########
    ##################################
    # 实例化 Tokenizer
    tokenizer_test = Tokenizer()
    text = "low lower new newest"

    # 设置词汇表的大小，训练tokenizer
    print("Training tokenizer-test...")
    tokenizer_test.train(text, vocab_size=2)
    print("Tokenizer-test trained successfully.")
    print("Vocabulary:", tokenizer_test.vocab)

    # 将字符串编码为token
    encoded = tokenizer_test.encode("newest")
    print(f"Encoded: {encoded}")

    # 解码回原始字符串
    decoded = tokenizer_test.decode(encoded)
    print(f"Decoded: {decoded}")


    ##################################
    #######  测试 manual文本  #########
    ##################################
    # 读取manual.txt文件
    with open("C:/Users/heifo/Desktop/Large-Language-Models-and-Alignment/hw1-code/bpe/ref/manual.txt", "r", encoding="utf-8") as f:
        manual_text = f.read()
    # print(manual_text)


    # 实例化 Tokenizer
    print("Initializing Tokenizer...")
    tokenizer = Tokenizer()

    # 设置词汇表的大小，训练 tokenizer
    print("Training tokenizer...")
    tokenizer.train(manual_text, vocab_size=1024)
    print("Tokenizer trained successfully.")
    # print("Vocabulary:", tokenizer.vocab)


    # 写入 tokenizer.vocab 到 vocab.txt
    with open("C:/Users/heifo/Desktop/Large-Language-Models-and-Alignment/hw1-code/bpe/ref/vocab.txt", "w", encoding="utf-8") as f:
        for word, freq in tokenizer.vocab.items():
            f.write(f"{word} {freq}\n")
            


    # # 将字符串编码为token
    # encoded = tokenizer.encode(manual_text)
    # # print(f"Encoded: {encoded}")
    # with open("C:/Users/heifo/Desktop/Large-Language-Models-and-Alignment/hw1-code/bpe/ref/manual_encoded.txt", "w", encoding="utf-8") as f:
    #     for token_id in encoded:
    #         f.write(f"{token_id}\n")


    # # 解码回原始字符串
    # decoded = tokenizer.decode(encoded)
    # # print(f"Decoded: {decoded}")
    # # 将编码结果保存到文件
    # with open("C:/Users/heifo/Desktop/Large-Language-Models-and-Alignment/hw1-code/bpe/ref/manual_decoded.txt", "w", encoding="utf-8") as f:
    #     f.write(" ".join(map(str, decoded)))


    ##################################
    #####  调用gpt2的tokenizer  #######
    ##################################
    # 加载huggingface transformers中的tokenizer
    from transformers import GPT2Tokenizer

    # 加载GPT-2的tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 定义句子
    sentence_1 = "Originated as the Imperial University of Peking in 1898, Peking University was China’s first national comprehensive university and the supreme education authority at the time. Since the founding of the People’s Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the “211 Project” and the “985 Project”, the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university."
    sentence_2 = "博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。"

    # 使用GPT-2的tokenizer进行编码
    gpt2_encoded_1 = gpt2_tokenizer.encode(sentence_1)
    gpt2_encoded_2 = gpt2_tokenizer.encode(sentence_2)

    # 输出编码后的结果
    print("############### GPT-2 Tokenizer Results ###############")
    print("############### Sentence 1 ###############")
    print(f"GPT-2 Tokenizer - Sentence 1: Length = {len(gpt2_encoded_1)}")
    print(f"GPT-2 Tokenizer - Sentence 1: Tokens = {gpt2_tokenizer.decode(gpt2_encoded_1)}")
    print(f"GPT-2 Tokenizer - Sentence 1 (Token IDs): {gpt2_encoded_1}")

    print("############### Sentence 2 ###############")
    print(f"GPT-2 Tokenizer - Sentence 2: Length = {len(gpt2_encoded_2)}")
    print(f"GPT-2 Tokenizer - Sentence 2: Tokens = {gpt2_tokenizer.decode(gpt2_encoded_2)}")
    print(f"GPT-2 Tokenizer - Sentence 2 (Token IDs): {gpt2_encoded_2}")