from openai import OpenAI
from textstat import textstat
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

client = OpenAI()

def evaluate_question(question_json, source_text):
    """
    Evaluates a question based on grade level, relevance, accuracy, and solution quality.

    Args:
        question_json (dict): A dictionary containing the question, options, correct_option, and solution.
        source_text (str): The source text to compare the question against.
        openai_api_key (str): OpenAI API key for GPT and embedding usage.

    Returns:
        dict: A dictionary containing the grade level, relevance score, accuracy result, and solution analysis.
    """
    


    # Extract question content
    question_content = question_json["question"]
    solution_content = question_json["solution"]

    # Step 1: Evaluate grade level
    grade_level = textstat.flesch_kincaid_grade(question_content)

    # Step 2: Check relevance using GPT embedding API
    def get_embedding(text, model="text-embedding-3-small"):
        response = client.embeddings.create(input=text, model=model).data[0]
        return np.array(response.embedding)

    question_embedding = get_embedding(question_content)
    source_embedding = get_embedding(source_text)
    relevance_score = cosine_similarity([question_embedding], [source_embedding])[0][0]

    # Step 3: Determine accuracy using GPT-4o mini
    prompt = f"Question: {question_json['question']}\nOptions: {', '.join(question_json['options'].values())}\nAnswer the question and provide the correct option letter ONLY."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
                ],
        max_tokens=50,
        temperature=0
    )
    gpt_answer = response.choices[0].message.content.strip()
    is_accurate = gpt_answer == question_json["correct_option"]

    # Step 4: Analyze solution explanation
    # Clarity: Use readability score
    solution_clarity = textstat.flesch_reading_ease(solution_content)

    # Completeness and Alignment: Use GPT to evaluate the explanation
    solution_prompt = (
        f"Question: {question_json['question']}\n"
        f"Correct Option: {question_json['correct_option']}\n"
        f"Solution Explanation: {solution_content}\n"
        f"Evaluate the explanation for clarity, completeness, and alignment with the correct answer."
    )
    solution_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": solution_prompt}
        ],
        max_tokens=100,
        temperature=0
    )
    solution_analysis = solution_response.choices[0].message.content.strip()

    # Calculate final overall score
    grade_level_score = max(0, 100 - abs(10 - grade_level) * 10)  # Penalize deviation from grade level 10
    relevance_score_scaled = relevance_score * 100  # Scale relevance score to 0-100
    accuracy_score = 100 if is_accurate else 0  # Full points for accuracy
    solution_quality_score = (solution_clarity / 100) * 50  # Scale clarity to 50 points
    overall_score = (
        0.3 * grade_level_score +
        0.3 * relevance_score_scaled +
        0.2 * accuracy_score +
        0.2 * solution_quality_score
    )

    # Return evaluation results
    return {
        "grade_level": grade_level,
        "relevance_score": relevance_score,
        "is_accurate": is_accurate,
        "solution_analysis": {
            "clarity": solution_clarity,
            "evaluation": solution_analysis
        },
        "overall_score": overall_score
    }
    
    
def get_stellar_evals():

      
      questions = open("stellar_questions.json", "r")
      questions = json.load(questions)
      source_text = open("source_text.txt", "r")
      
      source_text = source_text.read()
      
      results = []
      
      for question in questions["questions"]:
            eval = evaluate_question(question, source_text)
            print("Evaluating question: ", eval)
            results.append(eval)
            
      with open("stellar_evals.json", "w") as f:
            json.dump(results, f, indent=4)
            
if __name__ == "__main__":
      get_stellar_evals()
            
      
            
      