from flask import Flask, render_template
import pandas as pd
import pymongo
import requests

app = Flask(__name__)

client = pymongo.MongoClient(
    "mongodb+srv://mongoadmin:passwordone@cluster0.dgvwz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

hf_token = "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"
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

db = client.media
collection = db.social

# Sample user profile data
user_profile = {
    "username": "JohnDoe",
    "age": 30,
    "gender": "male",
    "dob": "2004-07-21T00:00:00.000+00:00",
    "interests": ["sports", "music", "travel"],
    "location": {
        "city": "Kolkata",
        "country": "India"
    }
}

# Sample similar profiles data
# similar_profiles = [
#     {"username": "JaneDoe", "age": 28, "gender": "female", "interests": ["sports", "music"]},
#     {"username": "AliceSmith", "age": 32, "gender": "female", "interests": ["travel", "reading"]}
#     # Add more similar profiles as needed
# ]

def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text})

    if response.status_code != 200:
        raise ValueError(
            f"Request failed with status code {response.status_code}: {response.text}")

    return response.json()

@app.route('/')
def index():
    query = "user having interest watching Movies"
    print("job completed")
    results = collection.aggregate([
        {   "$vectorSearch": {
                "queryVector": generate_embedding(query),
                "path": "rest_embedding",
                "numCandidates": 300,
                "limit": 5,
                "index": "vector_social_index",
            }
        }
    ]);

    similar_profiles = []

    for result in results:
        profile = {
            "name": result["Name"],
            "Gender": result["Gender"],
            "Interests": result["Interests"],
            "City": result["City"],
            "Country": result["Country"]
        }

        similar_profiles.append(profile)
    
    return render_template('index.html', user_profile=user_profile, similar_profiles=similar_profiles)


@app.route('/embedding/')
def embedding():
    data_URL =  "SocialMediaUsersDataset.csv" # path of the raw csv file

    review_df = pd.read_csv(data_URL)
    review_df.head()

    review_df = review_df['Interests'] + " " + review_df['City'] + " " + review_df['Country']
    review_df = review_df.sample(10)
    review_df["embedding"] = review_df.astype(str).apply(generate_embedding)

    # Make the index start from 0
    review_df.reset_index(drop=True)

    for doc in collection.find({'Interests': {"$exists": True}}):
        review_df = doc['Interests'] + " " + doc['City'] + " " + doc['Country']
        
        doc['rest_embedding'] = generate_embedding(review_df)
        collection.replace_one({'_id': doc['_id']}, doc)

    return "success"

if __name__ == '__main__':
    app.run(debug=True)
