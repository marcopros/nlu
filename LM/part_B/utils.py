import torch 
import torch.utils.data as data

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device:', DEVICE)

# Function to read a file and add an end-of-sentence (eos) token to each line
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            # Append each line with the eos token and strip any extra spaces
            output.append(line.strip() + " " + eos_token)
    return output

# Function to create a vocabulary from the given corpus
# Includes special tokens at the beginning of the vocabulary
def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0 
    # Add special tokens to the vocabulary first
    for st in special_tokens:
        output[st] = i
        i += 1
    # Add each word in the corpus to the vocabulary
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

# Class representing the language vocabulary
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        # Create word-to-id mapping and reverse id-to-word mapping
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}
    
    # Method to generate the vocabulary dictionary
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

# Custom dataset class for the Penn Treebank dataset
class PennTreeBank(data.Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        # Prepare source and target sequences by shifting the sentence
        for sentence in corpus:
            # Source: all words except the last
            self.source.append(sentence.split()[0:-1])
            # Target: all words except the first
            self.target.append(sentence.split()[1:])

        # Map words to their respective IDs
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        # Return the number of sequences in the dataset
        return len(self.source)

    def __getitem__(self, idx):
        # Get a specific data sample (source and target pair)
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary method to map sequences of words to sequences of IDs
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    # Handle out-of-vocabulary (OOV) words
                    print('OOV found!')
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res

# Function to merge a list of sequences into a batch with padding
def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        Merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Create a matrix filled with the pad token (0) with shape (batch_size, max_len)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            # Copy each sequence into the padded matrix
            padded_seqs[i, :end] = seq
        # Detach from the computational graph
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    # Sort data by sequence lengths in descending order
    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    # Organize data by key
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # Merge sources and targets
    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    # Move data to the specified device (GPU or CPU)
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    # Calculate the total number of tokens
    new_item["number_tokens"] = sum(lengths)
    return new_item