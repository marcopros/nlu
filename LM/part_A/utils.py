import torch 
import torch.utils.data as data

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'using device: {DEVICE}')    

# Reads a text file and adds an end-of-sentence token to each line
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Creates a vocabulary from a corpus, mapping words to unique IDs
def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0 
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

# Class to manage vocabulary and word-ID mappings
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    # Creates the vocabulary from a corpus and special tokens
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

# PyTorch dataset class for Penn TreeBank
class PennTreeBank(data.Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        # Prepare (source, target) word pairs from sentences
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) 
            self.target.append(sentence.split()[1:]) 
        
        # Map word sequences to IDs
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    # Returns a sample with source and target sequences
    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Maps each word in the sequence to its ID
    def mapping_seq(self, data, lang): 
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')  # Out-of-vocabulary (OOV) word
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res

# Collate function to batch and pad the sequences
def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        Merge from (batch * sentence_length) to (batch * max_length)
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)

        # Create a matrix filled with the PAD_TOKEN (0)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # Copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # Detach tensors from the computation graph
        return padded_seqs, lengths

    # Sort data by sequence length in descending order
    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # Merge source and target sequences with padding
    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    # Move tensors to GPU
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)  # Total number of tokens in the batch
    return new_item