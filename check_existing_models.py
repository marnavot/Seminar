from gensim.models import Word2Vec
import os
import chardet
import string

# import Seminar_year_graphs as gr
# Path to the directory containing your text files
# corpus_path = 'C:\\Users\\Tomer\\Documents\\עבודה סמינריונית\\wordLemPoS'
corpus_path = '/cs/usr/tomer.navot/Word_lemma_PoS'

# Set the path to save Word2Vec models
model_save_path = '/cs/labs/oabend/tomer.navot/decade_models'

for i in range(10):
    slot_path = f'{model_save_path}/{i}'
    # if not os.path.exists(slot_path):
    #     os.mkdir(slot_path)

    # Create a new Word2Vec model for each i (slot-division) if folder is empty
    if not any(os.scandir(slot_path)):
        model = Word2Vec(window=5, min_count=5, workers=4)

    slots = [(year, year + 10) for year in range(1800 + i, 2011, 10)]
    if i != 0:
        first_slot = (1800, 1800 + i)
        slots.insert(0, first_slot)
    print(slots)

    last_available_year = 1809

    for slot in slots:
        slot_model_path = os.path.join(slot_path, f'{slot[0]}-{slot[1] - 1}_model.model')
        if os.path.exists(slot_model_path):
            print(f"model of {slot_model_path} exists")
            last_available_year = slot[0]
            model = Word2Vec.load(slot_model_path)
            continue
        else:
            print(f"model of {slot_model_path} does not exist")