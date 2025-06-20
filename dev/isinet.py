import os
from os import listdir
from os.path import isfile, join
import pathlib
import shutil
import re
import ast
import math
import torch
from torch import nn
import pickle
import random
import time

def get_filenames(path):
    return [os.path.splitext(f)[0] for f in listdir(path) if isfile(join(path, f))]

def get_filepaths(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def create_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

class Attributes:
    def __init__(self, data_path, line_breaks = "\n"):
        self.data_path = data_path

        self.raw_path = f"{data_path}/raw"
        self.plain_path = f"{data_path}/plain"
        self.tokens_clean_path = f"{data_path}/tokens_clean"
        self.tokens_full_path = f"{data_path}/tokens_full"
        self.vocabularies_path = f"{data_path}/vocabularies"
        self.tokens_concepts_path = f"{data_path}/tokens_concepts"
        self.datasets_path = f"{data_path}/datasets"
        self.models_path = f"{data_path}/models"

        self.banned_path = f"{data_path}/banned.txt"

        self.line_breaks = line_breaks

        folders = os.getcwd().split('/')
        if (len(folders) == 1):
            folders = folders[0].split('\\')

        print(f"If your data_path is relative, it should be used from the current working directory: {os.getcwd()}")

    def init_data_dir(self):
        create_dir(self.raw_path)

        for item in os.listdir(self.data_path):
            item_path = os.path.join(self.data_path, item)
            
            # Skip the subdirectory itself to avoid recursion
            if os.path.abspath(item_path) == os.path.abspath(self.raw_path):
                continue
            
            target_path = os.path.join(self.raw_path, item)
            shutil.move(item_path, target_path)

        with open(os.path.join(self.data_path, "banned.txt"), 'w') as fp:
            pass

class Token():
    def __init__(self, text, is_clean):
        self.text = text
        self.is_clean = is_clean

class Dataset(torch.utils.data.Dataset):
    def __init__(self, load_file):
        with open(load_file, "rb") as lf:
            self.data = pickle.load(lf)
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

class SkipGram(nn.Module):
    def __init__(self, vocabulary, embed_size):
        super().__init__()
        self.central_embedding = nn.Embedding(num_embeddings=len(vocabulary)+1,
                                embedding_dim=embed_size, padding_idx=len(vocabulary))
        self.context_embedding = nn.Embedding(num_embeddings=len(vocabulary)+1,
                                embedding_dim=embed_size, padding_idx=len(vocabulary))

    def forward(self, center, contexts_and_negatives):
        v = self.central_embedding(center)
        u = self.context_embedding(contexts_and_negatives)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred
    
    def load(self, attrs, name):
        self.load_state_dict(torch.load(attrs.models_path + "/" + name + '.pt',weights_only=True))

class Process:
    def __init__(self, attrs):
        assert isinstance(attrs, Attributes)
        self.attrs = attrs

    def raw_to_plain(self, converter):

        raw_files = get_filepaths(self.attrs.raw_path)

        create_dir(self.attrs.plain_path)
        
        for f in raw_files:
            print(f"\033[94mConvirtiendo archivo: {f}\033[0m")

            content = converter(f"{self.attrs.raw_path}/{f}")
            
            with open(f"{self.attrs.plain_path}/{f}.txt", "ab") as t:
                t.write(content.encode("utf-8"))

    
    def tokenize(self, tokenizer):
        plain_files = get_filenames(self.attrs.plain_path)

        create_dir(self.attrs.tokens_full_path) 
        create_dir(self.attrs.tokens_clean_path)

        for f in plain_files:
            print("\033[94mTokenizando archivo: " + f + "\033[0m")

            with open(self.attrs.plain_path + "/" + f + ".txt", "rb") as pf:
                txt = pf.read().decode("utf-8")
                tokens = tokenizer(txt)
                with open(self.attrs.tokens_clean_path + "/" + f + ".txt", "wb") as tcf:
                    with open(self.attrs.tokens_full_path + "/" + f + ".txt", "wb") as tff:
                        for token in tokens:
                            enc_token = (token.text + self.attrs.line_breaks).encode("utf-8")
                            tff.write(enc_token)
                            if (token.is_clean):
                                tcf.write(enc_token)
    

    def get_candidate_concepts(self, window_size):

        tokens_clean_files = get_filenames(self.attrs.tokens_clean_path)

        candidates = {}

        banned_tokens = []

        with open(self.attrs.banned_path, "rb") as bf:
            banned_tokens = bf.read().decode("utf-8").split(self.attrs.line_breaks)

        for f in tokens_clean_files:
            print("\033[94mDetectando conceptos en archivo: " + f + "\033[0m")

            tokens = []

            with open(self.attrs.tokens_clean_path + "/" + f + ".txt", "rb") as tf:
                tokens = tf.read().decode("utf-8").split(self.attrs.line_breaks)

            for i in range(len(tokens) - window_size):
                window = tokens[i:i+window_size]
                
                arrays = [window[0:r] for r in range(1,window_size+1)]

                for arr in arrays:
                    arr = [s.lower() for s in arr]

                    if arr[-1] in banned_tokens:
                        break

                    if any(re.search(r'(^[0-9\.\,]+$)|(-$)|(^.\.$)|[0-9]{1,2}:[0-9]{1,2}|[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}', s) for s in arr):
                        continue

                    if len(arr) != len(set(arr)):
                        continue
                    
                    t = tuple(arr)
                    if (not t in candidates):
                        candidates[t] = 0
                    candidates[t] += 1
        return candidates


    def make_vocabs(self, candidates, window_size, freq_ranges):
        
        create_dir(self.attrs.vocabularies_path) 

        base = len(get_filenames(self.attrs.vocabularies_path))

        candidates_count = [0 for ws in range(window_size)]
        candidates_freqs_acum = [0 for ws in range(window_size)]
        for tokens, freq in candidates.items():
            candidates_count[len(tokens)-1] += 1
            candidates_freqs_acum[len(tokens)-1] += freq

        avg_candidates_freq = [candidates_freqs_acum[i] / candidates_count[i] for i in range(len(candidates_count))]


        for i, freq_range in enumerate(freq_ranges):
            print("\033[94mGenerando vocabulario: " + str(base + i) + " (" + str(freq_range) + ")\033[0m")
            vocabulary = []

            for tokens, freq in candidates.items():
                c = len(tokens)
                use_nth = c <= len(freq_range)
                ix = c-1 if use_nth else -1

                min_freq = freq_range[ix][0]
                max_freq = freq_range[ix][1]

                freq_rel_avg = freq / avg_candidates_freq[c-1]

                if (freq_rel_avg >= min_freq and freq_rel_avg <= max_freq):
                    vocabulary.append(tokens)

            with open(self.attrs.vocabularies_path + "/vocab_" + str(base + i) + ".txt", "wb") as cf:
                for concept in vocabulary:
                    cf.write((str(concept) + self.attrs.line_breaks).encode("utf-8"))
    
    

    def select_vocabulary(self, vocab_name):
        with open(self.attrs.vocabularies_path + "/" + vocab_name + ".txt", "rb") as cf:
            lines = cf.read().decode("utf-8").split("\n")[:-1]
            self.vocabulary = [ast.literal_eval(l) for l in lines]
        
        return len(self.vocabulary)
    

    def tokenize_by_concepts(self, window_size, window_size_extension_factor = 3):
        window_size_large = math.floor(window_size * window_size_extension_factor)

        create_dir(self.attrs.tokens_concepts_path)

        tokens_full_files = get_filenames(self.attrs.tokens_full_path)

        for f in tokens_full_files:
            print("\033[94mTokenizando por conceptos: " + f + "\033[0m")
            found_concepts = 0

            recent_concepts = {}

            with open(self.attrs.tokens_full_path + "/" + f + ".txt", "rb") as pf:
                tokens = pf.read().decode("utf-8").split("\n")
                tokens = [token.lower() for token in tokens]

                with open(self.attrs.tokens_concepts_path + "/" + f + ".txt", "wb") as tnf:

                    unks = 0

                    for i in range(len(tokens) - window_size_large):
                        window = tokens[i:i+window_size_large]
                        for k, v in recent_concepts.items():
                            if v > 0:
                                recent_concepts[k] -= 1

                        unks += 1
                    
                        for ix, concept in enumerate(self.vocabulary):
                            if (ix in recent_concepts and recent_concepts[ix] > 0):
                                continue

                            curr_word_ix = 0
                            curr_word = concept[curr_word_ix]

                            found = False

                            for token in window:
                                if(token == curr_word):
                                    curr_word_ix += 1
                                    if (len(concept) <= curr_word_ix):
                                        found = True
                                        break
                                    curr_word = concept[curr_word_ix]

                            if found:
                                if unks > 0:
                                    tnf.write(("-" + str(unks) + " ").encode("utf-8"))
                                    unks = 0
                                tnf.write((str(ix) + " ").encode("utf-8"))
                                recent_concepts[ix] = window_size_large
                                found_concepts += 1

    def make_datasets(self, window_sizes, K):

        def get_tokens_concepts_freqs():
            freq_abs = {}

            tokens_concepts_files = get_filenames(self.attrs.tokens_concepts_path)

            for file in tokens_concepts_files:
                with open(f"{self.attrs.tokens_concepts_path}/{file}.txt", "rb") as pf:
                    txt = pf.read().decode("utf-8")

                    nums = txt.split(" ") # Lista con cada número en el archivo
                    for num in nums:
                        if (num == "" or num[0] == "-"):
                            continue

                        token_ix = int(num)
                        if not token_ix in freq_abs:
                            freq_abs[token_ix] = 0

                        freq_abs[token_ix] += 1

            total_tokens = 0
            for token, freq in freq_abs.items():
                total_tokens += freq

            freq_rel = {}
            for token, freq in freq_abs.items():
                freq_rel[token] = freq/total_tokens

            return freq_abs, freq_rel

        class RandomGenerator:
            """Randomly draw among {1, ..., n} according to n sampling weights."""
            def __init__(self, sampling_weights):
                # Exclude
                self.population = list(range(1, len(sampling_weights) + 1))
                self.sampling_weights = sampling_weights
                self.candidates = []
                self.i = 0

            def draw(self):
                if self.i == len(self.candidates):
                # Cache `k` random sampling results
                    self.candidates = random.choices(
                        self.population, self.sampling_weights, k=10000)
                    self.i = 0
                self.i += 1
                return self.candidates[self.i - 1]

        data = [[] for i in range(0, len(window_sizes))]

        _, freq_rel = get_tokens_concepts_freqs()

        sampling_weights = [freq_rel[concept]**0.75 if concept in freq_rel else 0 for concept in range(0, len(self.vocabulary))]
        generator = RandomGenerator(sampling_weights)

        create_dir(self.attrs.datasets_path)
        
        tokens_concepts_files = get_filenames(self.attrs.tokens_concepts_path)
        window_sizes.sort()

        for file in tokens_concepts_files:
            print("\033[94mArmando dataset con: " + file + "\033[0m")

            with open(f"{self.attrs.tokens_concepts_path}/{file}.txt", "rb") as pf:
                txt = pf.read().decode("utf-8")

                nums = txt.split(" ") # Lista con cada número en el archivo

                for ix, num in enumerate(nums):
                    if (num == "" or num[0] == "-"):
                        continue

                    token_ix = int(num)

                    context = []
                    
                    curr_ws_ix = 0

                    c = 0
                    i = 1
                    
                    while curr_ws_ix < len(window_sizes): # Buscar conceptos hacia atrás
                        curr_ws = window_sizes[curr_ws_ix]
                        curr_wr = curr_ws // 2
                        c += curr_wr

                        context.append([])

                        while c >= 0: 
                            if (ix - i < 0): # Si se acabó el archivo, dejar de buscar
                                break
                            
                            val = nums[ix - i]
                            if (val == ""): 
                                val = "-0"
                            if (val[0] == "-"):
                                val = val.lstrip("-")
                                unks = int(val)
                                c = c - unks
                            else:
                                concept = int(val)
                                context[curr_ws_ix].append(concept)
                                c = c - 1
                            i = i + 1

                        curr_ws_ix += 1

                    curr_ws_ix = 0

                    c = 0
                    i = 1
                    l = len(nums)

                    while curr_ws_ix < len(window_sizes): # Buscar conceptos hacia adelante
                        curr_ws = window_sizes[curr_ws_ix]
                        curr_wr = curr_ws // 2
                        c += curr_wr

                        while c >= 0: 
                            if (ix + i < l): # Si se acabó el archivo, dejar de buscar
                                break
                            
                            val = nums[ix + i]
                            if (val == ""): 
                                val = "-0"
                            if (val[0] == "-"):
                                val = val.lstrip("-")
                                unks = int(val)
                                c = c - unks
                            else:
                                concept = int(val)
                                context[curr_ws_ix].append(concept)
                                c = c - 1
                            i = i + 1

                        curr_ws_ix += 1
                    
                    # Juntar contextos de tamaños de ventana más grandes con otros más pequeños
                    context_aux = []
                    context_acc = []

                    for sub_ctx in context:
                        context_acc += sub_ctx
                        context_aux.append(context_acc[:])
                    
                    context = context_aux

                    for ctx_ix, ctx in enumerate(context):
                        negatives = []
                        
                        while len(negatives) < len(ctx) * K:
                            neg = generator.draw()
                            if neg not in context:
                                negatives.append(neg)

                        if (len(ctx) > 0):
                            data[ctx_ix].append({
                                "center": token_ix,
                                "context": ctx,
                                "negatives": negatives
                            })
        
        for ws_ix, ws in enumerate(window_sizes):
            with open(self.attrs.datasets_path + "/dataset-" + str(ws) + ".pkl", "wb") as lf:
                pickle.dump(data[ws_ix], lf)

    def train_multiple(self, lr, num_epochs, embed_sizes, batch_sizes):
        datasets_files = get_filenames(self.attrs.datasets_path)

        create_dir(self.attrs.models_path)

        def collate_batch(data):
            max_len = max(len(d["context"]) + len(d["negatives"]) for d in data)
            centers, contexts_negatives, masks, labels = [], [], [], []
            for d in data:
                center = d["center"]
                context = d["context"]
                negative = d["negatives"]
                centers += [center]
                cur_len = len(context) + len(negative)
                contexts_negatives += [context + negative + [len(self.vocabulary)] * (max_len - cur_len)]
                masks += [[1] * cur_len + [0] * (max_len - cur_len)]
                labels += [[1] * len(context) + [0] * (max_len - len(context))]
            return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
                    contexts_negatives), torch.tensor(masks), torch.tensor(labels))
        
        class SigmoidBCELoss(nn.Module):
            # Binary cross-entropy loss with masking
            def __init__(self):
                super().__init__()

            def forward(self, inputs, target, mask=None):
                out = nn.functional.binary_cross_entropy_with_logits(
                    inputs, target, weight=mask, reduction="none")
                return out.mean(dim=1)

        loss = SigmoidBCELoss()

        def train(net, data_iter, lr, num_epochs, device):
            def init_weights(module):
                if type(module) == nn.Embedding:
                    nn.init.xavier_uniform_(module.weight)
            net.apply(init_weights)
            net = net.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            L = 0
            N = 0
            for epoch in range(num_epochs):
                start, num_batches = time.time(), len(data_iter)
                for i, batch in enumerate(data_iter):
                    optimizer.zero_grad()
                    center, context_negative, mask, label = [
                        data.to(device) for data in batch]

                    pred = net(center, context_negative)
                    l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                            / mask.sum(axis=1) * mask.shape[1])
                    l.sum().backward()
                    optimizer.step()
                    L += l.sum()
                    N += l.numel()
                timediff = time.time() - start
                if timediff == 0:
                    timediff = 1e-4
                print(f'loss {L / N:.3f}, '
                f'{N / (timediff):.1f} tokens/sec on {str(device)}')

        def try_gpu(i=0):
            if torch.cuda.device_count() >= i + 1:
                return torch.device(f'cuda:{i}')
            return torch.device('cpu')

        for f in datasets_files:
            ds = Dataset(self.attrs.datasets_path + "/" + f + ".pkl")
            for bs in batch_sizes:
                for es in embed_sizes:
                    print("\033[94mEntrenando modelo " + f + "-" + str(bs) + "-" + str(es) + "\033[0m")
                    dl = torch.utils.data.DataLoader(ds, bs, shuffle=True, collate_fn=collate_batch)
                    isinet = SkipGram(self.vocabulary, es)
                    train(isinet, dl, lr, num_epochs, try_gpu())
                    torch.save(isinet.state_dict(), self.attrs.models_path + "/" + f + "-" + str(bs) + "-" + str(es) + ".pt")
    

def get_related_concepts(vocabulary, model, concept_ix, k):
    W = model.central_embedding.weight.data
    x = W[torch.tensor(concept_ix)]

    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                    torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+2)[1].cpu().numpy().astype('int32')

    related = []
    left = k
    for i in topk[1:]:
        if k > 0 and i != len(vocabulary):
            related.append(vocabulary[i])
            k -= 1
    return related
