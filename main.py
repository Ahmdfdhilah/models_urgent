# api/main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from fastapi import FastAPI
from pydantic import BaseModel

# Baca dataset
data = pd.read_csv('../Dataset.csv')

# Pisahkan fitur dan label
X = data['Details']
y = data['Urgent']

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vektorisasi teks menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Buat dan latih model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Prediksi dan evaluasi model
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

app = FastAPI()

class Item(BaseModel):
    details: str

@app.post("/predict/")
async def predict(item: Item):
    # Vektorisasi input pengguna
    input_tfidf = vectorizer.transform([item.details])
    
    # Prediksi
    prediction = model.predict(input_tfidf)
    
    return {"urgent": bool(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
