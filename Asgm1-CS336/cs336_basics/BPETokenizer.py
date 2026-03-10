"""
Assignment 1
a. Implement BPE tokenizer
b. Implement Transformer, cross-entropy loss, AdamW optimizer, training loop
c. Train on TinyStories and OpenWebText
d. Leaderboard: minimize OpenWebText perplexity given 90 minutes on a H100
"""
import heapq
from collections import defaultdict


class BPETokenizer():

    def __init__(self):
        self.merge_rules = [] # tuple
        self.chunk_token = {}

        self.vocab_size = 0
        self.decode_table = defaultdict(str)
        self.encode_table = defaultdict(int)
    
    
    # ver3 pro
    def train(self, chunks, num_merges= 100):
        # 把token改为所有的pre tokenization的结果 改成word的freq
        # 进来的是 {word: freq}
        id_word = defaultdict(int) # {id: word}
        word_freq = defaultdict(int) # {id: freq}
        word_token = defaultdict(list) # {id: token}

        id = 0
        for word, freq in chunks.items():
            id_word[id] = word
            word_freq[id] = freq
            token = list(word)
            word_token[id] = token
            id += 1
    
        pair_freq = defaultdict(int)
        for id, token in word_token.items():
            for idx in range(len(token) - 1):
                pair = (token[idx], token[idx + 1])
                pair_freq[pair] += word_freq[id] # {("z","a"): int}

        while num_merges:
            num_merges -= 1
            
            pair_freq = defaultdict(int, {k: v for k, v in pair_freq.items() if v > 0})     # ✅：多出的维护步骤
            if not pair_freq:
                break

                    
            heap = []
            # 找到最大的pair, 维护一个最大堆
            for pair, freq in pair_freq.items():
                heapq.heappush(heap, (-freq, pair))

            # 找到最大的pair，刷新token，更新变量pair_freq, 同时后面用新的token来取代word_token
            max_pair = heap[0][1]

            for id, token in word_token.items():
                new_token = []
                if len(token) < 2:
                    continue
                # 正向查找是否匹配
                idx = 0
                while idx < len(token):
                    if idx < len(token) - 1 and (token[idx], token[idx + 1]) == max_pair:
                        # 修改max_pair影响的pair
                        current_freq = word_freq[id]
                        last_pair = new_last_pair = None
                        next_pair = new_next_pair = None

                        if idx > 0:
                            last_pair = (new_token[-1], token[idx]) # ✅：这里的new_token很有意思
                            new_last_pair = (new_token[-1], "".join(max_pair))
                        if idx + 2 < len(token):
                            next_pair = (token[idx + 1], token[idx + 2])
                            new_next_pair = ("".join(max_pair), token[idx + 2])

                        if last_pair:
                            pair_freq[last_pair] -= current_freq    
                            pair_freq[new_last_pair] += current_freq
                        if next_pair:
                            pair_freq[next_pair] -= current_freq
                            pair_freq[new_next_pair] += current_freq
                        
                        pair_freq[max_pair] -= current_freq

                        # 最后再放加入的merge
                        new_token.append("".join(max_pair))
                        idx += 2

                        self.merge_rules.append(max_pair)
                    else:
                        new_token.append(token[idx])
                        idx += 1
                
                # 修改word_token
                word_token[id] = new_token

        chunk_token = defaultdict(list)
        for id, token in word_token.items():
            chunk = id_word[id]
            chunk_token[chunk] = token

        self.chunk_token = chunk_token

        return chunk_token

    def build_vocab(self):
        # 建立vocab_id映射表
        vocabs= set()
        for tokens in self.chunk_token.values():
            for t in tokens:
                vocabs.add(t)

        # 建立映射表
        vocabs = sorted(list(vocabs))
        self.vocab_size = len(vocabs)

        for idx, vocab in enumerate(vocabs):
            self.encode_table[vocab] = idx # {vocab: idx}
            self.decode_table[idx] = vocab # {idx: vocab}

        print(f"词表构建完成，词表大小：{self.vocab_size}")
    
    def encode(self, text):
        # pre_tokenization -> list[pre] -> merge rules -> merged list -> encode
        pre_tokens = self.pretokenization(text) # ['lower','owner']

        # merge method 我觉得是encode部分的重点
        # 先使用chunk token来第一遍加工，对于不存在，在chunk里面的单词，则用merge规则合并
        encode_token = []
        for chunk in pre_tokens:
            if chunk in self.chunk_token:
                encode_token.extend(self.chunk_token[chunk])
            else:
                idx = 0
                while idx < len(chunk): # ['l','o',] [('l','k'),('l','o','w')]
                    # 对merge_rules的char开头的tuple进行长度排序
                    matching_rules = [rule for rule in self.merge_rules if rule[0] == chunk[idx]]
                    matching_rules.sort(key=len, reverse=True)

                    matched = False
                    for rule in matching_rules:
                        if chunk.startswith("".join(rule), idx):
                            encode_token.append(rule)
                            idx += len(rule)
                            matched = True
                            break
                    
                    if not matched:
                        encode_token.append("<unk_token>")
                        idx += 1
        
        encode_result = []
        for token in encode_token:
            encode_result.append(self.encode_table[token])

        return encode_result
    
    def decode(self, encode_result):
        
        encode_token = []
        for num in encode_result:
            encode_token.append(self.decode_table[num])


        return 

    def pretokenization(self, text):

        return 
        

    
 
# 统计了文本中单词的出现频率
chunks = {
    "low": 5,
    "lower": 2,
    "newest": 6,
    "widest": 3
}

# 实例化你的 Tokenizer 并运行
tokenizer = BPETokenizer()
# 假设我们只合并 10 次
result = tokenizer.train(chunks, num_merges=3)

for chunk, token in result.items():
    print(f"原词: {chunk:8} -> 拆解: {token}")