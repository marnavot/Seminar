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

# Function to read and preprocess the content of a file
def read_file(file_path):
    with open(file_path, 'rb') as file:
        # Detect the encoding of the file
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
        lines = file.readlines()
        # Extract lemma from the second column
        words = [line.split('\t')[1].lower() for line in lines if
                 len(line.split('\t')) > 1 and line.strip() and line.split('\t')[1].lower() not in string.punctuation]

        return words



for i in range(10):
    slot_path = f'{model_save_path}/{i}'
    # if not os.path.exists(slot_path):
    #     os.mkdir(slot_path)

    # Create a new Word2Vec model for each i (slot-division) if folder is empty
    if not any(os.scandir(slot_path)):
        model = Word2Vec(window=5, min_count=5, workers=4)

    slots = [(year, year + 10) for year in range(1810 + i, 2011, 10)]
    if i != 0:
        first_slot = (1810, 1810 + i)
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
            print(f"starting model of slot {slot}")
            slot_files = [os.path.join(corpus_path, file) for file in os.listdir(corpus_path) if
                            any(f"_{str(year)}_" in file for year in range(*slot))]

            # Read the files for the current year
            sentences = [read_file(file) for file in slot_files if os.path.isfile(file)]

            # Build or update the vocabulary
            if model.wv.key_to_index:
                model.build_vocab(sentences, update=True)
            else:
                model.build_vocab(sentences)

            # Check if the model has an existing vocabulary
            if not model.wv.key_to_index:
                # If the model has no prior vocabulary, continue to the next slot
                print(f"{slot[0]}-{slot[1]}: no vocab")
                continue

            # Train the model for the current slot
            model.train(sentences, total_examples=model.corpus_count, epochs=1)

            # Save the model for the current slot
            model.save(slot_model_path)
            print(f"{slot_model_path} saved")

# for i in range(10):
#     slots = [(year, year + 10) for year in range(1800 + i, 2011, 10)]
#     if i != 0:
#         first_slot = (1800, 1800 + i)
#         slots.insert(0, first_slot)
#     slot_path = f'C:\\Users\\Tomer\\PycharmProjects\\pythonProject\\decade_models\\{i}'
#     slot_models = {slot[0]:
#                        Word2Vec.load(os.path.join
#                                      (slot_path, f'{slot[0]}-{slot[1] - 1}_model.model'))
#                    for slot in slots}
#     vectors = gr.get_vector_dict(gr.top_100, models=slot_models)
#     gr.make_all_year_graphs(gr.top_100_vectors, folder=slot_path)