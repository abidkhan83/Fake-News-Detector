from flask import Flask, request, render_template
import re
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


MODEL_PATH = "model.h5"
TOKENIZER_PATH = "tokenizer.pickle"
MAXLEN = 1000 

# Load trained model and tokenizer
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

# Build the model
model.build(input_shape=(None, MAXLEN))
print(model.summary())

# Flask app
app = Flask(__name__)

# Preprocessing 
def clean_text(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # remove special chars
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    text = text.strip()
    text = text.lower() 
    return text

# Prediction function
def predict_fake_news(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])

    # Debugging 
    print("🔎 Cleaned text:", cleaned)
    print("🔎 Tokenized sequence:", seq)

    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
    print("🔎 Padded shape:", padded.shape)
    print("🔎 First 20 tokens:", padded[0][:20])

    pred = model.predict(padded, verbose=0)[0][0]

    if pred >= 0.5:
        return f"REAL NEWS ({pred*100:.2f}%)"
    else:
        return f"FAKE NEWS ({(1-pred)*100:.2f}%)"

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        input_text = request.form["news_text"]
        result = predict_fake_news(input_text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
