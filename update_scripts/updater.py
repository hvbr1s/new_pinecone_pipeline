import time
import json
import os
from dotenv import load_dotenv
import openai
from tqdm.auto import tqdm
import os
import pinecone
from openai.error import RateLimitError

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data

def run_updater(json_file_path:str = None, index_name = 'hc'):
    if not json_file_path:
        pinecone_pipeline_root_directory = os.path.dirname(os.path.dirname(__file__))
        output_folder = os.path.join(pinecone_pipeline_root_directory, 'output_files')
        json_file_path = os.path.join(output_folder, 'output.json')
    documents = read_json_file(json_file_path)

    texts = [doc['text'] for doc in documents]


    # initialize connection to pinecone
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENVIRONMENT')
    )


    # connect to index
    index = pinecone.Index(index_name)

    # define embed model
    embed_model = "text-embedding-ada-002"

    batch_size = 100  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(documents), batch_size)):
        # find end of batch
        i_end = min(len(documents), i+batch_size)
        meta_batch = documents[i:i_end]
        # get texts to encode
        texts = [x['text'] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(input=texts, engine=embed_model)
        except RateLimitError:
            # OpenAI's API has a cooldown time of 1 minute if a rate limit is hit.
            time.sleep(61)
            # If it fails a second time, raise the error and quit
            res = openai.Embedding.create(input=texts, engine=embed_model)
        embeds = [record['embedding'] for record in res['data']]
        # format the vectors to upsert
        to_upsert = [{'id': meta['id'], 'values': embed, 'metadata': meta} for meta, embed in zip(meta_batch, embeds)]
        # upsert to Pinecone
        index.upsert(vectors=to_upsert)

    print('Database updated!')

if __name__ == "__main__":
    run_updater()