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
from flask_swagger_ui import get_swaggerui_blueprint
from flask_swagger import swagger



app = Flask(__name__)
app.config.from_object(Config)
pc = Pinecone(api_key = app.config['PINECONE_API_KEY'])
index_name = app.config['VECTOR_INDEX_NAME']
spec = ServerlessSpec(cloud="AWS", region="us-east-1")


client = OpenAI(
    api_key=app.config['OPENAI_API_KEY']
)
openai.api_key=app.config['OPENAI_API_KEY']
MODEL = "text-embedding-3-large"


# Define a route to serve the Swagger specification as a JSON response
@app.route('/api/spec', methods=['GET'])
def spec():
    base_url = request.url_root.rstrip('/')
    https_url = "https://" + base_url.split('://')[1]
    http_url = "http://" + base_url.split('://')[1]
    swagger_spec = {
        "openapi": "3.0.1",
        "info": {
                "title": "Enhanced Chat API",
                "description": "Enhanced Chat API with create and insert embeddings, perform search, enhance the completions API",
                "version": "1.0"
                },
         "servers": [
                        {
                        "url": https_url #"http://192.168.0.107:5000/"
                        },
                        {
                        "url": http_url #"https://192.168.0.107:5000/"
                        }
                    ],
        "paths": {
                "/delete_index": {
                        "delete": {
                        "summary": "Delete the index",
                        "description": "Index successfully deleted.",
                        "parameters": [
                                        {
                                            "name": "index_name",
                                            "in": "header",
                                            "description": "Name of the index to be deleted",
                                            "required": True,
                                            "schema": {
                                            "type": "string"
                                            }
                                        }
                                     ],
                        "responses": {
                        "200": {
                            "description": "Successfull deletion",
                            "content": {
                            "*/*": {
                                "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {
                                    "type": "string",
                                    "example": "The Index is successfully deleted."
                                    }
                                }
                                }
                            }
                            }
                        }
                        }
                    }
                },
                "/embeddings/generate": {
                "post": {
                    "summary": "Generate embeddings and insert to pinecone database",
                    "description": "Success message after successfull insert.",
                    "requestBody": {
                    "content": {
                        "application/x-www-form-urlencoded": {
                        "schema": {
                            "required": [
                            "file_path",
                            "index_name"
                            ],
                            "type": "object",
                            "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path of the file from which data is read."
                            },
                            "index_name": {
                                "type": "string",
                                "description": "Name of the index to which data is stored."
                            }
                            }
                        }
                        }
                    },
                    "required": True
                    },
                    "responses": {
                    "200": {
                        "description": "Successfull Insert",
                        "content": {
                        "*/*": {
                            "schema": {
                            "type": "object",
                            "properties": {
                                "message": {
                                "type": "string",
                                "example": "Successfull Insert"
                                }
                            }
                            }
                        }
                        }
                    },
                    "500": {
                        "description": "API Response",
                        "content": {
                        "*/*": {
                            "schema": {
                            "type": "object",
                            "properties": {
                                "message": {
                                "type": "string",
                                "example": "Failed to insert"
                                }
                            }
                            }
                        }
                        }
                    }
                    }
                }
                },
                "/enhance_prompt": {
                "post": {
                    "summary": "Get response after searching pinecone and enhancing the prompt",
                    "description": "Response after sucessfull Completions call.",
                    "requestBody": {
                    "content": {
                        "application/x-www-form-urlencoded": {
                        "schema": {
                            "required": [
                            "prompt"
                            ],
                            "type": "object",
                            "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "question to be answered."
                            }
                            }
                        }
                        }
                    },
                    "required": True
                    },
                    "responses": {
                    "200": {
                        "description": "API Response",
                        "content": {
                        "*/*": {
                            "schema": {
                            "type": "object",
                            "properties": {
                                "message": {
                                "type": "string",
                                "example": "API Response"
                                }
                            }
                            }
                        }
                        }
                    },
                    "500": {
                        "description": "API Response",
                        "content": {
                        "*/*": {
                            "schema": {
                            "type": "object",
                            "properties": {
                                "message": {
                                "type": "string",
                                "example": "Failed to retrieve"
                                }
                            }
                            }
                        }
                        }
                    }
                    }
                }
            }
        },
        "components": {}
    }
    return jsonify(swagger_spec)

SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/api/spec'  # Our API url (can of course be a local resource)

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Enhanced Chat"
    },
)

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

#Generate Embeddings
def generate_embeddings_from_text(text,index_name):
    try:
        indx = index_exits(index_name)
         
        # Call the OpenAI embeddings API to generate embeddings
        response = client.embeddings.create(input=text, model=MODEL)

        # Extract and return the embeddings
        embeddings = response.data[0].embedding
        meta = {'text': text}
        # upsert to Pinecone
        
        to_upsert = {
            "id": str(round(random.random(), 2)), 
            "values": embeddings, 
            "metadata": meta
            }
        indx.upsert(vectors=[to_upsert])
        print("Embeddings_Text--->",indx.describe_index_stats())
        return embeddings

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
        
        # Call the OpenAI embeddings API to generate embeddings
        response = client.embeddings.create(input=text, model=MODEL)

        # Extract and return the embeddings
        embeddings = response.data[0].embedding
        meta = {'text': text}
        # upsert to Pinecone
        print(indx.describe_index_stats())
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
                    paragraph += line  
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
            
            #print("ids_batch---->",ids_batch)
        
            # create embeddings
            res = client.embeddings.create(input=paragraphs[i], model=MODEL)
            #embeds = [record.embedding for record in res.data]
            embeds = res.data[0].embedding
            #print("embeds--------------------------------->",embeds)
            
            #print(str(ids_batch[i]))
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
        
        res = indx.query(vector=[xq], top_k=1, include_metadata=True)

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

        print("Contexts------------->",contexts)
        if contexts:
            combined_prompt = f'{prompt} {" ".join(contexts)}'
            flag = 'Y'
        else:
            combined_prompt = prompt
            flag = 'N'
        # Request payload
        payload = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": combined_prompt,
            "temperature": 0.7,
            "max_tokens": 1000
        }

        # Make the request
        response = requests.post(url, headers=headers, json=payload)

        # Parse response
        if response.status_code == 200:
            text = response.json()['choices'][0]['text']
            print("Completions response---->",text)
            print("Flag--->",flag)
            if flag == 'N':
                generate_embeddings_from_text(text,index_name)
            return response.json()['choices'][0]['text'].strip()
        else:
            print("Error:", response.text)
            return None
    
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
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

app.register_blueprint(swaggerui_blueprint)

if __name__ == '__main__':
    app.run(debug=True)

