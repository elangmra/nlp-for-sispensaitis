from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

@app.route('/', methods=['POST'])
def score_essay():
    data = request.json

    if not data or 'student_answer' not in data or 'key_answer' not in data:
        return jsonify({"error": "Data input tidak lengkap"}), 400

    student_answer = data['student_answer']
    key_answer = data['key_answer']

    if not student_answer or not key_answer:
        return jsonify({"error": "Jawaban siswa atau kunci jawaban kosong"}), 400

    # TF-IDF dan Cosine Similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([student_answer, key_answer])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2]).flatten()[0]

    # Kalkulasi skor akhir (hanya menggunakan cosine similarity)
    final_score = cosine_sim

    return jsonify({"score": final_score})

if __name__ == '__main__':
    app.run(debug=True)
