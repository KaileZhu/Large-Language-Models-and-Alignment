# 读取文件内容
with open('/media/shuozhang/PKUWorking/Large Language Models and Alignment/hw1-code/bpe/manual.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 持续替换 '\n\n' 为 '\n'，直到不再有 '\n\n'
while '\n\n' in content:
    content = content.replace('\n\n', '\n')

# 写回文件
with open('/media/shuozhang/PKUWorking/Large Language Models and Alignment/hw1-code/bpe/test.txt', 'w', encoding='utf-8') as file:
    file.write(content)