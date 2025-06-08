from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import socket
import os
import time

app = Flask(__name__)  # 🚀 აპლიკაციის შექმნა
THRESHOLD = 0.49  # 49% სიმპათია, შეგიძლია ითამაშო ამაზე! 🎯

# 👀 უხილავი სწავლა ჩართვა/გამორთვა
INVISIBLE_LEARNING_MODE = True

# 🌍 გლობალური ვარიაბლები თემისა და სტატუსისთვის
last_topic = None
last_response_time = 0
chat_done = False

# 🧠 ტექსტის წინასწარი დამუშავება
def preprocess_text(text):
    text = text.lower()  # 🔡 ტექსტის ციფრები და დიდი ასოები მცირე ასოებად
    text = re.sub(r'[^\w\s]', '', text)  # 🧹 სპეციალური სიმბოლოები გამოტოვე
    return text

# ❓ გაურკვეველი კითხვების შენახვა
def save_unknown_question(question_text):
    try:
        with open("shenaxuli.txt", "a", encoding="utf-8") as f:
            f.write(f"Q: {question_text.strip()}\nA: BLANK\n")  # ✍️ შეკითხვა და პასუხი
        print(f"[LOG] შეუმჩნეველი კითხვა შენახულია: {question_text.strip()}")  # 📝
    except Exception as e:
        print(f"[ERROR] კითხვა ვერ შეინახა: {e}")  # 🚨

# 🔎 საუკეთესო პასუხის გამოტანა
def get_combined_response(user_input, questions, answers, vectorizer, question_vectors, threshold=THRESHOLD):
    parts = re.split(r'[?.,;]\s*|\sდა\s', user_input.lower())  # 🤔 ტექსტის გაყოფა ნაწილებად
    matched_answers = []  # 📋 პასუხების ლისტი
    unknown_flag = True  # ❓ ჯერ უცნობია

    # 🔍 ვეძებთ საუკეთესო პასუხს
    for part in parts:
        sub_parts = [p.strip() for p in re.split(r'\sან\s', part) if p.strip()]
        temp_answers = []  # 🧩 დროებითი პასუხები

        for sub in sub_parts:
            sub_clean = preprocess_text(sub)
            if not sub_clean:
                continue  # 🚫 ცარიელი ტექსტის გამოტოვება
            user_vec = vectorizer.transform([sub_clean])  # 🔄 ტექსტის ვექტორიზაცია
            similarity = cosine_similarity(user_vec, question_vectors)[0]  # 🧠 მსგავსი კითხვების პოვნა
            best_idx = np.argmax(similarity)  # 🥇 საუკეთესო პასუხის მოძებნა
            if similarity[best_idx] >= threshold:
                possible_answers = answers[best_idx]
                selected_answer = random.choice(possible_answers)  # 🎲 შემთხვევითი პასუხი
                temp_answers.append(selected_answer)
                unknown_flag = False

        if len(sub_parts) > 1 and temp_answers:
            choice = random.choice([  # 🎯 სხვადასხვა ვარიანტი
                random.choice(temp_answers),
                ' და '.join(temp_answers),
                ', '.join(temp_answers) + '.'
            ])
            matched_answers.append(choice)
        else:
            matched_answers.extend(temp_answers)

    if matched_answers:
        return ' '.join(matched_answers).strip()  # 🎤 პასუხი
    else:
        save_unknown_question(user_input)  # 📥 შეუმჩნეველი კითხვა
        return "ვწუხვარ, ეს ვერ გავიგე. სცადე სხვანაირად კითხვა."  # ❌ პასუხი ვერ მოიძებნა

# 📚 FAQ მონაცემების დატვირთვა
def load_faq_data():
    global faq_data, questions, answers, vectorizer, question_vectors

    file_list = []
    combined_text = ""  # 📝 ფაილების შერწყმა

    try:
        with open("database.txt", 'r', encoding='utf-8') as f:
            file_list = [line.strip() for line in f if line.strip()]  # 📄 database.txt წაკითხვა
    except Exception as e:
        print("database.txt წაკითხვა ვერ მოხერხდა:", e)  # 🚨
        return [], [], [], None, None

    # 🗑️ 'learned.txt' ამოიღე
    # 'learned.txt' აღარ გამოიყენება!

    # 👨‍💻 სხვა ფაილების გადაკითხვა
    for filename in file_list:
        if not os.path.exists(filename):
            print(f"ფაილი '{filename}' ვერ მოიძებნა.")  # 🔍 ფაილი ვერ მოიძებნა
            continue
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                combined_text += f.read() + "\n"  # 🗂️ ტექსტის შერწყმა
        except Exception as e:
            print(f"ფაილის '{filename}' წაკითხვა ვერ მოხერხდა: {e}")  # 🚨

    faq_data = {}
    question = None  # 🤔 შეკითხვის საწყისი
    for line in combined_text.split('\n'):
        line = line.strip()
        if line.startswith("Q:"):
            question = line[2:].strip()  # ❓ კითხვა
        elif line.startswith("A:") and question:
            answer = line[2:].strip()  # 💬 პასუხი
            if question not in faq_data:
                faq_data[question] = []
            faq_data[question].append(answer)  # 📥 პასუხის შენახვა
            question = None

    questions = list(faq_data.keys())  # 🔑 შეკითხვების აღება
    answers = [faq_data[q] for q in questions]  # 💡 პასუხების ჩამონათვალი
    processed_questions = [preprocess_text(q) for q in questions]  # 🧹 შეკითხვების წინასწარი დამუშავება

    if processed_questions:
        vectorizer = TfidfVectorizer().fit(processed_questions)  # 📊 ვექტორიზაცია
        question_vectors = vectorizer.transform(processed_questions)
    else:
        vectorizer = None
        question_vectors = None

    return faq_data, questions, answers, vectorizer, question_vectors

faq_data, questions, answers, vectorizer, question_vectors = load_faq_data()  # 📥 FAQ მონაცემების დატვირთვა

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')  # 🖥️ მთავარ გვერდზე გადამისამართება

@app.route('/chat', methods=['POST'])
def chat():
    global last_topic, last_response_time, chat_done, faq_data, questions, answers, vectorizer, question_vectors

    data = request.json
    if not data or 'message' not in data:
        return jsonify({'response': "შეცდომა: არ არის შეტყობინება"}), 400  # ❌ შეცდომა

    if vectorizer is None or question_vectors is None or question_vectors.shape[0] == 0:
        return jsonify({'response': "ჩატბოტი ჯერ არ არის კონფიგურირებული."}), 500  # 🧠 არ არის კონფიგურირებული

    user_message = data['message'].strip()  # ✍️ შეტყობინება

    start_time = time.time()  # ⏳ დროის აღება
    response_text = get_combined_response(user_message, questions, answers, vectorizer, question_vectors)  # 🧠 პასუხის გამოთვლა
    end_time = time.time()

    elapsed = end_time - start_time  # ⏱️ დროის გამოთვლა
    print(f"[LOG] პასუხის გამოთვლის დრო: {elapsed:.2f} წამი")  # ⏳

    if elapsed <= 3:
        chat_done = False
    else:
        chat_done = True

    last_response_time = elapsed  # 🕒 დრო

    print(f"[LOG] მომხმარებლის შეტყობინება: {user_message}")  # 💬
    print(f"[LOG] პასუხი: {response_text}")  # 📄
    print(f"[LOG] ჩატი შედგა: {chat_done}")  # 🏁

    return jsonify({
        'response': response_text,  # 💬 პასუხი
        'chat_done': chat_done  # 🏁
    })

# 🔌 თავისუფალი პორტის მოძებნა
def find_open_port(start_port=8080, max_attempts=10):
    port = start_port
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                port += 1
    return None

if __name__ == "__main__":
    port = find_open_port(8080)  # 🔍 პორტის მოძებნა
    if port is None:
        print("არ მოიძებნა თავისუფალი პორტი 8080-დან 8090-მდე.")  # 🚨
    else:
        print(f"სერვერი გაშვებულია პორტზე: {port}")  # 🚀
        app.run(host='0.0.0.0', port=port, debug=True)  # 🏃‍♂️ სერვერის გაშვება
