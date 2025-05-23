from flask import Flask, request, jsonify, render_template
from lime.lime_text import LimeTextExplainer
import numpy as np
import shap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import traceback

from utils import (
    clean_text,
    preprocess_for_word2vec_BiLSTM,
    predict_using_w2v_bilstm
)

app = Flask(__name__)

# Load the trained BiLSTM model
bilstm_model = load_model("my_model.h5")

class_names = ['Fake', 'Real']
explainer = LimeTextExplainer(class_names=class_names)


def explain_prediction(article_text):
    def predict_proba(texts):
        results = []
        for text in texts:
            cleaned = clean_text(text)
            vectors, _ = preprocess_for_word2vec_BiLSTM(cleaned)
            # vectors = pad_sequences([vectors], maxlen=300, dtype='float32', padding='post', truncating='post')
            pred = predict_using_w2v_bilstm(bilstm_model, vectors)
            prob = float(pred)
            results.append([1 - prob, prob])
        return np.array(results)

    explanation = explainer.explain_instance(article_text, predict_proba, num_features=10)
    return explanation.as_list()


def generate_detailed_explanation(explanation_list, label):
    top_words = sorted(explanation_list, key=lambda x: abs(x[1]), reverse=True)[:5]
    insights = []
    for word, weight in top_words:
        insight = {
            "word": word,
            "impact": weight,
            "reason": "commonly found in fake news" if weight < 0 else "frequently used in reliable articles"
        }
        insights.append(insight)
    if label == "Fake":
        summary = (
            "The model suspects this article might be fake due to terms like "
            + ", ".join(f'\"{i["word"]}\"' for i in insights[:3])
            + " that often appear in misleading content."
        )
    else:
        summary = (
            "This article appears trustworthy thanks to words like "
            + ", ".join(f'\"{i["word"]}\"' for i in insights[:3])
            + ", typically seen in credible sources."
        )
    return summary, insights


# SHAP Explanation
def explain_with_shap(vectors, tokens, model):
    MAX_SEQUENCE_LEN = 150
    EMBEDDING_DIM = 100

    # padding للمقالة الواحدة (vectors)
    padded = np.zeros((1, MAX_SEQUENCE_LEN, EMBEDDING_DIM), dtype='float32')
    length = min(len(vectors), MAX_SEQUENCE_LEN)
    if length > 0:
        padded[0, :length, :] = vectors[:length]

    # نكررها 10 مرات كـ background
    background = np.repeat(padded, 10, axis=0)  # shape (10, 150, 100)

    # التفسير باستخدام SHAP
    explainer = shap.GradientExplainer(model, background)

    shap_values = explainer.shap_values(padded)  # shape (1, 150, 100)

    word_shap_pairs = []
    for i in range(min(len(tokens), len(shap_values[0][0]))):
        impact = float(np.linalg.norm(shap_values[0][0][i]))
        word_shap_pairs.append((tokens[i], impact))


    return sorted(word_shap_pairs, key=lambda x: abs(x[1]), reverse=True)



@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        article = data.get("article", "")
        print("Received article:", article)

        cleaned = clean_text(article)
        print("Cleaned article:", cleaned)

        vectors, tokens = preprocess_for_word2vec_BiLSTM(cleaned)
        #vectors = pad_sequences([vectors], maxlen=300, dtype='float32', padding='post', truncating='post')

        print("Vectors shape:", vectors.shape, "Tokens:", tokens[:5])

        prediction = predict_using_w2v_bilstm(bilstm_model, vectors)
        prediction_value = float(prediction)
        label = "Real" if prediction_value > 0.5 else "Fake"

        shap_insights = explain_with_shap(vectors, tokens, bilstm_model)

        top_words = shap_insights[:5]
        word_insights = [{
            "word": word,
            "impact": impact,
            "reason": "commonly found in fake news" if impact < 0 else "frequently used in reliable articles"
        } for word, impact in top_words]

        if label == "Fake":
            human_explanation = (
                "The model suspects this article might be fake due to terms like "
                + ", ".join(f'\"{i["word"]}\"' for i in word_insights[:3])
                + " that often appear in misleading content."
            )
        else:
            human_explanation = (
                "This article appears trustworthy thanks to words like "
                + ", ".join(f'\"{i["word"]}\"' for i in word_insights[:3])
                + ", typically seen in credible sources."
            )

        return jsonify({
            "prediction": prediction_value,
            "label": label,
            "shap_insights": shap_insights,
            "human_explanation": human_explanation,
            "insights": word_insights
        })

    except Exception as e:
        print("❌ Error in /predict:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)
