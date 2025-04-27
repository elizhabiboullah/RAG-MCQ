import os
import json
import requests
from tqdm import tqdm

def validate():
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(current_dir, "data", "processed", "5_estate_planning_questions.json")

    print(f"Trying to open file at: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # FLATTEN the questions
    question_list = []
    for chapter_questions in questions.values():
        question_list.extend(chapter_questions)

    correct = 0
    total = len(question_list)

    # Loop for validation
    for q in tqdm(question_list, desc="Validating Questions"):
        payload = {
            "question": q["question"],
            "options": q.get("options", [])
        }

        try:
            response = requests.post("http://localhost:8000/predict", json=payload)
            response.raise_for_status()  # Raise error if status code is not 200
        except requests.exceptions.RequestException as e:
            print(f"Request failed for question: {q['question']}\nError: {e}")
            continue

        response_data = response.json()

        # Debugging: print response from API
        print(f"Server response for question: {q['question']}")
        print(response_data)

        if "predicted_answer" not in response_data:
            print(f"Error: No 'predicted_answer' key in response for question: {q['question']}")
            continue  #if there's no predicted answer

        prediction = response_data["predicted_answer"].strip().upper()
        correct_answer = q["answer"].strip().upper()  # Use "answer" instead of "expected_answer"

        # Check if the predicted matches
        if prediction == correct_answer:
            correct += 1
        else:
            print(f"Wrong: Q: {q['question']}\nPredicted: {prediction}, Correct: {correct_answer}\n")

    # Calculate and print accuracy
    accuracy = correct / total * 100
    print(f"Validation complete. Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    validate()
