from transformers import GPT2Tokenizer

# 加载GPT-2的tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 定义句子
sentence_1 = "Originated as the Imperial University of Peking in 1898, Peking University was China’s first national comprehensive university and the supreme education authority at the time. Since the founding of the People’s Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the “211 Project” and the “985 Project”, the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university."
sentence_2 = "博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。"

# 使用GPT-2的tokenizer进行编码
gpt2_encoded_1 = gpt2_tokenizer.encode(sentence_1)
gpt2_encoded_2 = gpt2_tokenizer.encode(sentence_2)

print

# 输出编码后的结果
print("############### GPT-2 Tokenizer Results ###############")
print(f"GPT-2 Tokenizer - Sentence 1: {sentence_1}")
print(f"GPT-2 Tokenizer - Sentence 1: Length = {len(gpt2_encoded_1)}")
print(f"GPT-2 Tokenizer - Sentence 1: Tokens = {gpt2_tokenizer.decode(gpt2_encoded_1)}")
print(f"GPT-2 Tokenizer - Sentence 1 (Token IDs): {gpt2_encoded_1}")

print(f"GPT-2 Tokenizer - Sentence 2: {sentence_2}")
print(f"GPT-2 Tokenizer - Sentence 2: Length = {len(gpt2_encoded_2)}")
print(f"GPT-2 Tokenizer - Sentence 2: Tokens = {gpt2_tokenizer.decode(gpt2_encoded_2)}")
print(f"GPT-2 Tokenizer - Sentence 2 (Token IDs): {gpt2_encoded_2}")