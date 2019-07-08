from gensim.models import KeyedVectors
import pandas as pd
import pdb

# run after sourcing virtual env


def load_entities(csv_file):
    try:
        labels = pd.read_csv(csv_file)
        return [labels.iloc[idx,0] for idx in xrange(len(labels))]
    except IOError as e:
        print 'Could not load csv file:' + str(csv_file)
        raise IOError


if __name__ == "__main__":
    w2v = KeyedVectors.load_word2vec_format('./google_w2v.bin',binary=True)
    entities = load_entities('sd_thor_mp3d_entities.csv')
    e2v = {}
    for entity in entities:
        w2v_query_word = entity.split('.')[0]
        vector_not_found = True
        while vector_not_found:
            try:
                e2v[entity] = w2v[w2v_query_word]
                vector_not_found = False
            except KeyError as e:
                print e
                w2v_query_word = raw_input('Please type a similar word.\n')

    with open('w2v_thor_mp3d_entity_embeddings.csv','w') as f:
        for entity in entities:
            for i in xrange(len(e2v[entity])):
                f.write(str(e2v[entity][i])+' ')
            f.write('\n')
    pdb.set_trace()
