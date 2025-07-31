from flask import Flask, render_template, request
from transformers import pipeline
import torch
app = Flask(__name__)

# Function to chunk the passage into smaller parts
def chunk_text(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if request.method == "POST":
        passage = request.form.get("passage")
        question = request.form.get("question")
        chunk_size = request.form.get("chunk-size", 400)
        
        if not passage:
            return render_template("dashboard.html", error="Please enter a passage.")
        if not question:
            return render_template("dashboard.html", error="Please enter a question.")
        
        chunk_size = int(chunk_size)
        passage_chunks = chunk_text(passage, chunk_size)

        qna_pipeline = pipeline("question-answering",framework="pt",
                                model="bert-large-uncased-whole-word-masking-finetuned-squad",
                                tokenizer="bert-large-uncased")

        answer = ""
        highlighted_chunks = []

        for idx, chunk in enumerate(passage_chunks):
            result = qna_pipeline({"question": question, "context": chunk})
            if idx == 0:
                answer = result["answer"]
            else:
                answer += " " + result["answer"]
            if result["score"] > 0.1:
                highlighted_chunks.append(chunk)

        return render_template("dashboard.html", answer=answer, highlighted_chunks=highlighted_chunks)

    return render_template("dashboard.html")
if __name__ == "__main__":
    app.run(debug=True)