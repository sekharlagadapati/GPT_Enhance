import requests
import random
import time
import traceback
import openai
from flask import Flask, request, jsonify
from openai import APIError, Completion
from openai import OpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec
from datasets import load_dataset
from tqdm.auto import tqdm
from config import Config


app = Flask(__name__)
app.config.from_object(Config)
pc = Pinecone(api_key = app.config['PINECONE_API_KEY'])
index_name = app.config['VECTOR_INDEX_NAME']
spec = ServerlessSpec(cloud="GCP", region="Iowa (us-central1)")


client = OpenAI(
    api_key=app.config['OPENAI_API_KEY']
)
openai.api_key=app.config['OPENAI_API_KEY']
MODEL = "text-embedding-3-large"


""" res = client.embeddings.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], model=MODEL
)
#print(res)

# we can extract embeddings to a list
embeds = [record.embedding for record in res.data]
len(embeds)

print(len(embeds)) """
# function to check if index already exists (it shouldn't if this is your first run)
def index_exits(index_name):
    if index_name not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=len("3072"),  # dimensionality of text-embed-3-large
            metric='dotproduct',
            spec=spec
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # connect to index
    index = pc.Index(index_name)
    time.sleep(1)
    # view index stats
    index.describe_index_stats()
    
    
    print(index.describe_index_stats())
    return index
""" # load the first 1K rows of the TREC dataset
trec = load_dataset('trec', split='train[:100]')
print(trec)
count = 0  # we'll use the count to create unique IDs
batch_size = 32  # process everything in batches of 32
 for i in tqdm(range(0, len(trec['text']), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(trec['text']))
    # get batch of lines and IDs
    lines_batch = trec['text'][i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = client.embeddings.create(input=lines_batch, model=MODEL)
    embeds = [record.embedding for record in res.data]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert)) """
 
 # Function to read file contents and generate embeddings

@app.route('/embeddings/generate', methods=['POST'])
def generate_embeddings():
    try:
        data = request.json
        file_path = data.get('file_path')
        index_name = data.get('index_name')
        
        embeddings = generate_embeddings_from_file_paragraphs(file_path, index_name)

        #embeddings = generate_embeddings_from_file(file_path, index_name)
        
        if embeddings:
            return jsonify({'status': 'Successfully upserted to pinecode database'}), 200
        else:
            return jsonify({'error': 'Failed to generate embeddings'}), 500
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return None    
#Generate Embeddings by reading from a file
def generate_embeddings_from_file(file_path,index_name):
    try:
        indx = index_exits(index_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(file_path)
        #print(text)
        # Call the OpenAI embeddings API to generate embeddings
        response = client.embeddings.create(input=text, model=MODEL)

        #print(response)
        # Extract and return the embeddings
        #embeddings = response['embedding']
        embeddings = response.data[0].embedding
        #print(embeddings)
        meta = {'text': text}
        #print(meta)
        #to_upsert = zip(round(random.randint(1, 100)), embeddings, meta)
        # upsert to Pinecone
        #index.upsert(vectors=list(to_upsert))
        print(indx)
        to_upsert = {
            "id": str(round(random.random(), 2)), 
            "values": embeddings, 
            "metadata": meta
            }
        indx.upsert(vectors=[to_upsert])
        return embeddings

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return None

#Generate Embeddings by reading each scentence from a file
def generate_embeddings_from_file_paragraphs(file_path,index_name):
    paragraphs = []
    indx = index_exits(index_name)
    try:
        with open(file_path, 'r') as file:
            paragraph = ''
            
            for line in file:
                if line.strip():  # Check if the line is not empty
                    # Split the line into sentences.
                   
                    sentences = line.split(".")
                    for sentence in sentences:
                        if sentence.strip():
                            paragraph += sentence
                            
                            if paragraph:
                                paragraphs.append(paragraph.strip())  # Remove leading/trailing whitespace
                                paragraph = ''  # Reset paragraph for the next iteration
                    else:
                        if paragraph:
                            paragraphs.append(paragraph.strip())  # Remove leading/trailing whitespace
                            paragraph = ''  # Reset paragraph for the next iteration
            if paragraph:  # Handle the last paragraph if file doesn't end with an empty line
                paragraphs.append(paragraph.strip())
        print("Paragraph Length ------------------------->" , len(paragraphs))
        #print(paragraphs)
        i=0
        length = len(paragraphs)
        #print(length)
        for i in range(0, len(paragraphs)):
             # set end position of batch
            i_end = length
            print(paragraphs[i])
            print("----------------------------------------")
            # get batch of lines and IDs
            #lines_batch = [paragraphs[n] for n in range(i, i_end)]
            ids_batch = [str(n) for n in range(0, i_end+1)]
            print("ids_batch---->",ids_batch)
        
            # create embeddings
            res = client.embeddings.create(input=paragraphs[i], model=MODEL)
            #embeds = [record.embedding for record in res.data]
            embeds = res.data[0].embedding
            #print("embeds--------------------------------->",embeds)
            """ # prep metadata and upsert batch
            #meta = {}
            meta = [{"text": paragraphs[i]}]
            print("meta--->", meta)
            to_upsert = zip(ids_batch,embeds, meta )
            #to_upsert = zip(to_upsert,{'text': paragraphs[i]})
            print("to_upsert--------->",list(to_upsert))
            # upsert to Pinecone
            indx.upsert(vectors=list(to_upsert))  """
            print(str(ids_batch[i]))
            meta = {"text": paragraphs[i]}
            to_upsert = {
            "id": str(ids_batch[i]), 
            "values": embeds, 
            "metadata": meta
            }
            #print("to_upsert--------->",to_upsert)
            indx.upsert(vectors=[to_upsert])
            return embeds
        
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return None


#embeds = generate_embeddings_from_file("C:\\Users\\chand\\GPT_Vector\\Test.txt",index)
#generate_embeddings_from_file_paragraphs("C:\\Users\\chand\\GPT_Vector\\Test.txt",index)

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    index_name = data.get('index_name')
    results = {}
    results = search_in_pinecone(query, index_name)
    
    results_text = []
    for match in results['matches']:
        results_text[match] = results.match['metadata']['text']
        
    print("results in API---->", results_text)
    if results:
        return jsonify(results), 200
    else:
        return jsonify({'error': 'Failed to perform search'}), 500

#Function to search pinecode
def search_in_pinecone(query,index):
    try: 
        indx = index_exits(index)
    
        prompt = query
        #query = "Unicode: One encoding standard for many alphabets"
        xq = client.embeddings.create(input=prompt, model=MODEL).data[0].embedding
        
        res = indx.query(vector=[xq], top_k=5, include_metadata=True)

        print(indx.describe_index_stats())
        """ for match in res['matches']:
            print(f"{match['score']:.2f}: {match['metadata']['text']}") """
        print("Results------------->",res.matches)
        return res
    except Exception as e:
        print(f"Error Searching Index: {e}")
        traceback.print_exc()
        return None

@app.route('/delete_index', methods=['DELETE'])
def delete_index():
    index_name = request.args.get('index_name')
    
    deleted = delete_index(index_name)
    
    if deleted:
        return jsonify({'message': 'Index deleted successfully'}), 200
    else:
        return jsonify({'error': 'Failed to delete index'}), 500


#Function to delete a index from Pinecode
def delete_index(index_name):
    pc.delete_index(index_name)

# Function to generate completions using rest api calls
def get_chat_completion(prompt,contexts):
    try:
        # API endpoint
        url = 'https://api.openai.com/v1/completions'

        # Request headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + openai.api_key
        }
        combined_prompt = f'{prompt} {" ".join(contexts)}'
        # Request payload
        payload = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": combined_prompt,
            "temperature": 0.7,
            "max_tokens": 300
        }

        # Make the request
        response = requests.post(url, headers=headers, json=payload)

        # Parse response
        if response.status_code == 200:
            return response.json()['choices'][0]['text'].strip()
        else:
            print("Error:", response.text)
            return None
    
    except Exception as e:
        print("Error:", e)
        return None

# Function to generate completions using OpenAI Completions API
def generate_completions(prompt, contexts):
    # Combine prompt and contexts
    combined_prompt = f'{prompt} {" ".join(contexts)}'
    prompt = {"role": "user", "content":combined_prompt}
    try:
    # Generate completions using OpenAI Completions API
        completions = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[prompt],
            max_tokens=300
        )
        return completions.choices
    except APIError as e:
        print(f"Error generating completions: {e}")
        traceback.print_exc()
        return None


@app.route('/enhance_prompt', methods=['POST'])
def enhance_prompt():
    data = request.json
    prompt = data.get('prompt')
    
    enhanced_completions = enhance_prompt(prompt)
    #enhanced_completions = enhanced_completions[0].message
    
    if enhanced_completions:
        return jsonify(enhanced_completions), 200
    else:
        return jsonify({'error': 'Failed to enhance prompt'}), 500

# Example usage
def enhance_prompt(prompt):
    try:
        # Search for similar prompts in Pinecone
        search_results = search_in_pinecone(prompt,index_name)
        
        search_results = search_results.matches
        print(search_results)
        # Extract contexts from search results
        contexts = [search_results['metadata']['text'] for search_results in search_results]
        #contexts = [contexts.text for contexts in contexts]
        # Generate completions using combined prompt and contexts
        #completions = generate_completions(prompt, contexts)
        completions = get_chat_completion(prompt, contexts)

        return completions

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return None

""" # Example usage
prompt = "Unicode: One encoding standard for many alphabets"
enhanced_completions = enhance_prompt(prompt)
print(enhanced_completions) """

if __name__ == '__main__':
    app.run(debug=True)

#generate_embeddings_from_file_paragraphs("C:\\Users\\chand\\GPT_Vector\\Test.txt",index_name)
#generate_embeddings_from_file("C:\\Users\\chand\\GPT_Vector\\Test.txt",index_name)

