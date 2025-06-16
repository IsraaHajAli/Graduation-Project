import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string
from ftfy import fix_text
from tf_explain.core.integrated_gradients import IntegratedGradients
import tensorflow as tf
from tf_keras_vis.saliency import Saliency
import spacy  # ✅ لإضافة spaCy
import os

from captum.attr import IntegratedGradients


# تحميل spaCy model مرة واحدة (إذا مش محمل عندك)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# إعدادات
max_len = 150
embedding_dim = 100

# تحميل DistilBERT
tokenizer = DistilBertTokenizerFast.from_pretrained("Ensemble_Files/distilbert_saved")
bert_model = DistilBertForSequenceClassification.from_pretrained("Ensemble_Files/distilbert_saved")
bert_model.eval()

# تحميل BiLSTM + Word2Vec
bilstm_model = load_model("Ensemble_Files/my_bilstm_model.h5")
w2v_model = Word2Vec.load("Ensemble_Files/word2vec_model.bin")

# تحميل أدوات NLTK
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))



explanatory_reasons = {
    "fake": "Suggests the content may be fabricated or misleading.",
    "hoax": "Used to label something as intentionally deceptive.",
    "scam": "Indicates manipulation or fraud.",
    "bullshit": "Slang expressing disbelief or falsehood.",
    "propaganda": "Implies agenda-driven or biased messaging.",
    "agenda": "Suggests hidden motives behind the content.",
    "lie": "Attacks truthfulness of the content.",
    "manipulated": "Indicates altered or deceptive information.",
    "twisted": "Suggests intentional distortion.",
    "damn": "Emotional or informal language, weakens credibility.",
    "wtf": "Slang; emotionally reactive and informal tone.",
    "liar": "Discredits source by attacking honesty.",
    "trash": "Derogatory term implying low quality or falsehood.",
    "sheeple": "Used to describe blindly obedient people; emotional.",
    "you won’t believe": "Clickbait language, often in fake articles.",
    "wake up": "Populist language urging disbelief of mainstream views.",
}

trusted_reasons = {
    "confirmed": "Shows official or verified information.",
    "officials": "Implies formal authority or governmental source.",
    "report": "Implies info is based on documented data.",
    "reportedly": "Suggests indirect but formal reporting.",
    "study": "Suggests scientific or academic support.",
    "research": "Backed by investigation or experimentation.",
    "published": "Appeared in recognized source or medium.",
    "evidence": "Sign of supporting facts.",
    "source": "Indicates reference to original information.",
    "percent": "Use of statistics gives credibility.",
    "figures": "Numerical data enhances objectivity.",
    "statistics": "Indicates data-based reasoning.",
    "agency": "Implies an organized, possibly governmental body.",
    "government report": "Highly formal and institutional data.",
    "peer-reviewed": "Endorsed by academic community.",
    "experts": "Shows reliance on authoritative opinion.",
    "ministry": "Indicates government-level source.",
    "spokesperson": "Represents official statements.",
    "journal": "Suggests scientific publication.",
    "UN": "Global institutional authority.",
    "WHO": "World Health Organization; trusted medical source.",
    "NATO": "Military and political alliance; trusted geopolitical info.",
    "data shows": "Grounds the statement in measurable facts.",
    "validated": "Confirmed through formal process.",
    "survey": "Data collected systematically.",
    "transcript": "Verbatim record, ensures accuracy.",
    "records": "Historical or factual documentation.",
    "historical data": "Backed by long-term records.",
    "scholars": "Academic authority.",
    "committee": "Formal group decision or investigation.",
    "panel": "Group of experts; formal evaluation.",
}


# تنظيف النص
def clean_text(text):
    text = fix_text(text)
    text = re.sub("\[.*?\]", " ", text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text.lower()


# توقع باستخدام DistilBERT
def predict_distilbert(text):
    inputs = tokenizer(
        text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[0][1].item()


# توقع باستخدام BiLSTM
def predict_bilstm(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words
    ]
    vecs = []
    for word in tokens[:max_len]:
        if word in w2v_model.wv:
            vecs.append(w2v_model.wv[word])
        else:
            vecs.append(np.zeros(embedding_dim))
    while len(vecs) < max_len:
        vecs.append(np.zeros(embedding_dim))
    vecs = np.array(vecs).reshape(1, max_len, embedding_dim)
    prob = bilstm_model.predict(vecs)[0][0]
    return prob

###############################################################################################################################################
############################################################# Ensemble Prediction #############################################################
###############################################################################################################################################

# توقع إجمالي من النموذجين
def ensemble_predict(text):
    text = clean_text(text)
    prob_bert = predict_distilbert(text)
    prob_bilstm = predict_bilstm(text)
    avg_prob = (prob_bert + prob_bilstm) / 2
    prediction = 1 if avg_prob > 0.5 else 0
    print(
        f"🧠 DistilBERT: {prob_bert:.3f}, 🌀 BiLSTM: {prob_bilstm:.3f}, ✅ Average: {avg_prob:.3f}"
    )
    return prediction



################################################################################################################################################################
############################################################# XAI Functions [BiLSTM with Word2Vec] #############################################################
################################################################################################################################################################

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words
    ]
    return tokens


def vectorize(tokens):
    vec = []
    for word in tokens:
        if word in w2v_model.wv:
            vec.append(w2v_model.wv[word])
        else:
            vec.append(np.zeros(embedding_dim))
    while len(vec) < max_len:
        vec.append(np.zeros(embedding_dim))
    return np.array(vec[:max_len]).reshape(1, max_len, embedding_dim)


def extract_phrases(text):
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]


def explain_bilstm_with_sentences_and_phrases(text):
    saliency = Saliency(bilstm_model)

    def score_function(output):
        return output

    results = []

    # تحليل الجمل
    sentences = sent_tokenize(text)
    print(f"\n🔍 Found {len(sentences)} sentences")
    for sentence in sentences:
        tokens = preprocess_text(sentence)
        if not tokens:
            continue
        sentence_input = vectorize(tokens)
        saliency_map = saliency(score_function, sentence_input)
        score = np.sum(np.abs(saliency_map))
        results.append(("Sentence", sentence.strip(), score))

    # تحليل العبارات (phrases)
    phrases = extract_phrases(text)
    print(f"🧠 Found {len(phrases)} phrases")
    for phrase in phrases:
        tokens = preprocess_text(phrase)
        if not tokens:
            continue
        phrase_input = vectorize(tokens)
        saliency_map = saliency(score_function, phrase_input)
        score = np.sum(np.abs(saliency_map))
        results.append(("Phrase", phrase.strip(), score))

    results = sorted(results, key=lambda x: x[2], reverse=True)
    

    # فلترة الجمل التي تحتوي على كلمات مشبوهة
    filtered_explanatory = []
    for typ, content, score in results:
        if any(k in content.lower() for k in explanatory_reasons):
            filtered_explanatory.append((typ, content, score))

    # فلترة الجمل التي تحتوي على كلمات موثوقة
    filtered_trusted = []
    for typ, content, score in results:
        if any(k in content.lower() for k in trusted_reasons):
            filtered_trusted.append((typ, content, score))

    # تحديد التصنيف النهائي للمقال
    predicted_label = ensemble_predict(text)

    if predicted_label == 1:
        print("\n🔷 Trusted & Formal Evidence Found (Real Articles):\n")
        if filtered_trusted:
            for typ, content, score in filtered_trusted[:10]:
                print(f"- [{typ}] {content} (Score: {score:.4f})")
                for word in trusted_reasons:
                    if word in content.lower():
                        print(f"Reason: {trusted_reasons[word]}\n")
                        break
                print()

                
        else:
            print("⚠️ No strong formal evidence detected.")
    else:
        print("\n🔥 Top Interpretable Units (Fake/Misleading Articles):\n")
        if filtered_explanatory:
            for typ, content, score in filtered_explanatory[:10]:
                print(f"- [{typ}] {content} (Score: {score:.4f})")
                for word in explanatory_reasons:
                    if word in content.lower():
                        print(f"Reason: {explanatory_reasons[word]}\n")
                        break
                print()



######################################################################################################################################################
############################################################# XAI Functions [DistilBERT] #############################################################
######################################################################################################################################################


def explain_distilbert(text):
    # تجهيز التوكنز والتأكد إنها LongTensor
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs['input_ids'].to(torch.long)
    attention_mask = inputs['attention_mask'].to(torch.long)

    # تعريف الدالة اللي راح نستخدمها مع Captum
    def forward_func(input_ids, attention_mask):
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[:, 1]  # class 1: Fake

    # إنشاء الكائن الخاص بالـ Integrated Gradients
    ig = IntegratedGradients(forward_func)

    # تنفيذ الإرجاع مع الانتباه للمتحكمات الإضافية
    attributions = ig.attribute(input_ids, additional_forward_args=(attention_mask,), target=1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_scores = attributions[0].sum(dim=-1).detach().numpy()

    print("\n🧠 Top Influential Tokens:")
    pairs = list(zip(tokens, token_scores))
    pairs = sorted(pairs[1:-1], key=lambda x: abs(x[1]), reverse=True)  # remove [CLS], [SEP]

    for token, score in pairs[:10]:
        print(f"{token:<12} | Score: {score:.4f}")
        for k in explanatory_reasons:
            if k in token.lower():
                print(f"⚠️  Explanatory: {k} → {explanatory_reasons[k]}")
        for k in trusted_reasons:
            if k in token.lower():
                print(f"✅ Trusted: {k} → {trusted_reasons[k]}")
        print()


#################################################################################################################################################################
############################################################# Testing on a sample to try the models #############################################################
#################################################################################################################################################################


# ✅ مثال اختبار
text1 = (
    "WASHINGTON (Reuters) - U.S. Interior Secretary Ryan Zinke on Thursday launched an effort to reduce U.S. dependence on "
    "foreign supplies of critical minerals used in smartphones, computers and military equipment, which he said poses a national "
    "security and economic risk. Under a directive from President Donald Trump, Zinke will  work with Defense Secretary Jim Mattis "
    "to publish in 60 days a list of non-fuel minerals that are vulnerable to supply chain disruptions and necessary for manufacturing"
    " and will develop a strategy to lessen U.S. dependence on foreign suppliers. The policy would aim to identify new domestic "
    "sources of critical minerals; increase domestic exploration, mining and  recycling; giving miners and producers electronic "
    "access to better mapping and geological data; and streamlining leasing and permitting for new mines. “The United States must "
    "not remain reliant on foreign competitors like Russia and China for the critical minerals needed to keep our economy strong and "
    "our country safe,” Trump said. The order comes after the Interior Department and the U.S. Geological Survey published a report "
    "earlier this week that detailed U.S. dependence on foreign competitors for its supply of certain minerals. "
    "The report identified 23 out of 88 minerals that are priorities for U.S. national defense and the economy because they are "
    "components in products ranging from batteries to military equipment. The list included rare earths metals, lithium, graphite "
    "and other minerals.  That report did not offer policy recommendations, but Zinke said he would rely on the findings as he "
    "prioritizes research into certain mineral deposit areas on federal land and plans policies to promote mining. Twenty of the 23 "
    "critical minerals that the United States relies on are sourced from China. Much of the world’s lithium is produced in Australia "
    "and Chile, with the bulk of the world’s reserves straddling huge salt flats in the so-called lithium triangle of Chile, Bolivia "
    "and Argentina.  Lithium exports from Chile, for example, approached $600 million in 2016, or roughly 40 percent of the global "
    "market by volume, according to Chile development agency Corfo. Lithium producers SQM, Albemarle and FMC Lithium are among the "
    "region’s top producers."
)

text = (
    "The recent spike in fuel prices has left many citizens frustrated and confused. While officials claim it's due to global supply chain issues and increased demand, some folks are starting to call bullshit on the whole explanation. A few outspoken commentators on social media have suggested that it's just another scam by big oil companies to rake in record profits while pretending to care about the environment."
    "“I mean, come on,” one user tweeted. “They’ve been screwing us over for decades. Now they slap the word 'green' on everything and expect us to swallow the same crap all over again. Wake up!”"
    "Others were more measured in their responses, citing reports from credible sources like the International Energy Agency and independent watchdogs. But still, the narrative of corporate greed and manipulation continues to spread, especially among online forums where users freely throw around terms like “propaganda,” “hoax,” and “agenda.”"
    "The media’s role in this has also been called into question. Some alternative news outlets have accused mainstream networks of cherry-picking data to fit their stories, while ignoring any dissenting views. One writer even described the coverage as “twisted trash that insults our intelligence.”"
    "Is this just another cycle of overreaction and misinformation, or is there a deeper pattern of manipulation at play? Either way, the public is angry, confused, and desperate for someone to just tell the damn truth."
)


label = ensemble_predict(text)
print("Final Prediction:", "Real ✅" if label == 1 else "Fake 🚫")
print("\n\n\n")

#explain_bilstm_with_sentences_and_phrases(text)
explain_distilbert(text)


