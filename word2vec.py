from gensim.models import Word2Vec

corpus_file_path = '/home/luizaperez/Downloads/uniprot_kmers.random_sample.txt'

model = Word2Vec(corpus_file=corpus_file_path, vector_size=100, window=5, min_count=1, workers=4, epochs=30)

model.save('word2vec.model')

selec = input('Selecione o token: ')

palavras = print(model.wv.most_similar(selec))
modelo = print(model.wv[selec])

model.save('/home/luizaperez/Documentos/word2vec/word2vec.model')
