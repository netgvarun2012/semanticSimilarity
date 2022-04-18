# import libraries
from fastapi import FastAPI,Path
from pydantic import BaseModel
from sentence_transformers import  util
from pyngrok import ngrok
import uvicorn
import nest_asyncio
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
model.load_state_dict(torch.load('pytorch_model.bin'), strict=False)

app = FastAPI()

class SentencePairs(BaseModel):
  sent1: str
  sent2: str

@app.get("/")
def read_root():
    return {"test_response": "The API is working!"}


def calcsemanticscore(s1,s2):
  ret_dict = {}
  #Compute embedding for both lists
  embeddings1 = model.encode(s1, convert_to_tensor=True)
  embeddings2 = model.encode(s2, convert_to_tensor=True)

  #Compute cosine-similarits
  cosine_scores = util.cos_sim(embeddings1, embeddings2)[0][0]
  print(f'cosine score is {cosine_scores}')
  #Output the pairs with their score
  ret_dict['similarity score'] = round(float(cosine_scores),2)

  return ret_dict

@app.post("/semanticdetector/")
async def analyse_text(sentPairs: SentencePairs):
  d = {}
  d = calcsemanticscore(sentPairs.sent1,sentPairs.sent2)
  print(f'd is {d}')
  return d
