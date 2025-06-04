import collections
from collections import Counter
import random
import math

class Vocab:
    """Vocabulario para texto."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Cuenta las frecuencias de los tokens
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # La lista de tokens únicos
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        # Mapea cada token a su índice
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
         # Retorna el tamaño del vocabulario
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
         # Convierte un índice o una lista de índices en sus tokens correspondientes
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self): # Índice para el token desconocido
        return self.token_to_idx['<unk>']

def make_vocab(oraciones, min_freq=1):
    if oraciones and isinstance(oraciones[0], list):
        tokens = [token for line in oraciones for token in line]
    else:
        tokens = oraciones

    vocabulario = Vocab(tokens, min_freq=min_freq)
    counter_obj = Counter(tokens)
    sorted_by_freq = sorted(counter_obj.items(), key=lambda x: x[1], reverse=True)
    return vocabulario, sorted_by_freq

with open('./data/tokens.txt', 'r', encoding='utf-8') as f:
    oraciones = [line.strip() for line in f if line.strip()]

# Crear vocabulario
vocab_completo, _ = make_vocab(oraciones)
vocab_min_freq_10, ordenado = make_vocab(oraciones, min_freq=10)

print("Vocabulario sin frecuencia mínima:", len(vocab_completo))
print("Vocabulario con min_freq=10:", len(vocab_min_freq_10))


def subsample(oraciones):
    """Subsample high-frequency words de una lista de listas de tokens."""
    # Si las oraciones son strings, las tokenizamos
    #if oraciones and isinstance(oraciones[0], str):
    #    oraciones = [line.split() for line in oraciones]

    # Aplanar la lista de listas a una sola lista para contar frecuencias
    # tokens = [token for line in oraciones for token in line]
    tokens = oraciones

    counter_obj = collections.Counter(tokens)
    num_tokens = sum(counter_obj.values())

    def keep(token):
        # Probabilidad de mantener el token según su frecuencia
        return random.uniform(0, 1) < math.sqrt(1e-4 / (counter_obj[token] / num_tokens))

    # Crear nueva lista de oraciones con tokens muestreados (subsampled)
    # oraciones_subsampled = [[token for token in line if keep(token)] for line in oraciones]
    oraciones_subsampled = [token for token in tokens if keep(token)]

    return oraciones_subsampled, counter_obj

oraciones_subsampled, counter = subsample(oraciones)

print(f"Original tokens: {sum(len(line) for line in [o.split() for o in oraciones])}")
print(f"Tokens after subsampling: {sum(len(line) for line in oraciones_subsampled)}")
print(f"Ten more frequent tokens:")
for token, freq in counter.most_common(10):
    print(f"{token}: {freq}")