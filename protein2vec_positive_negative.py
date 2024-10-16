from gensim.models import Word2Vec
import numpy as np
import csv
import pandas as pd
import glob

model = Word2Vec.load('/home/luizaperez/Documentos/word2vec/word2vec.model')

# gera os kmers das proteínas
def generate_kmers(sequence, k):
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

# converte kmers em vetores
def kmer2vec(kmers, model):
    vectors = [model.wv[i] for i in kmers if i in model.wv]
    return vectors

# média do vetor das proteínas
def protein2vec(sequence, k, model):
    if isinstance(sequence, str) and sequence.strip():
        sequence = sequence.upper()
        kmers = generate_kmers(sequence, k)
        vectors = kmer2vec(kmers, model)
        if vectors:
            mean_vector = np.mean(vectors, axis=0)
            return mean_vector
    return None

# escreve os vetores em um csv
def write_vector_to_csv(writer, sequence, vectors):
    if vectors is not None:
        writer.writerow([sequence, vectors.tolist()])
    else:
        writer.writerow([sequence, 'Nenhum vetor válido foi encontrado para essa proteína'])

diretorio = '/home/luizaperez/data/raw/*.csv'
k = 5

# arquivos output sequencias positivas e negativas
with open('/home/luizaperez/Documentos/word2vec/vetores_positivos.csv', mode='w', newline='') as pos_file, \
     open('/home/luizaperez/Documentos/word2vec/vetores_negativos.csv', mode='w', newline='') as neg_file:

    pos_writer = csv.writer(pos_file)
    neg_writer = csv.writer(neg_file)

    pos_writer.writerow(['Sequência', 'Result', 'Vetor médio'])
    neg_writer.writerow(['Sequência', 'Result', 'Vetor médio'])
    
    for file in glob.glob(diretorio):
        df = pd.read_csv(file)
        
        for _, row in df.iterrows():
            sequence = row['sequence']
            result = row['result']
            
            if pd.notna(sequence) and sequence.strip():
                vectors = protein2vec(sequence, k, model)
                if vectors is not None:
                    if result.lower() in ['positive', 'positive-low', 'positive-high', 'positive-intermediate']:
                        pos_writer.writerow([sequence, result, vectors.tolist()])
                    elif result.lower() == 'negative':
                        neg_writer.writerow([sequence, result, vectors.tolist()])
                else:
                    if result.lower() in ['positive', 'positive-low' 'positive-high', 'positive-intermediate']:
                        pos_writer.writerow([sequence, result, 'Nenhum vetor válido encontrado'])
                    elif result.lower() == 'negative':
                        neg_writer.writerow([sequence, result, 'Nenhum vetor válido encontrado'])

model.save('/home/luizaperez/Documentos/word2vec/prot2vec_positivo_negativo.model')
