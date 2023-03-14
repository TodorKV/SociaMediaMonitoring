from fastapi import FastAPI
from pydantic import BaseModel
import mk_model as model
import snscrape.modules.twitter as sntwitter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Msg(BaseModel):
    msg: str


@app.get("/")
async def root():
    return {"message": 'hey'}

@app.get("/{count}")
async def count_get(count: int):
    vectoriser, LRmodel = model.load_models()
    if count > 100:
        count = 100
    query = "(Бугарија OR бугарија)"
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i > count:  # only retrieve count tweets
            break
        tweets.append(tweet.rawContent)

    return {"tweets": model.predict(vectoriser, LRmodel, tweets)}