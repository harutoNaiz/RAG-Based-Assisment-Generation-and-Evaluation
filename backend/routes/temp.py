from flask import Blueprint, request, jsonify
import os
from utils import extract_number, get_stripped_name
from config import get_db
from groq import Groq
import pandas as pd
import re
from dotenv import load_dotenv
import faiss
from sentence_transformers import SentenceTransformer
# from external_data import ExternalDataFetcher   

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

# Initialize sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

temp_start_bp = Blueprint('temp_start_bp', __name__)
csv_generation_bp = Blueprint('csv_generation_bp', __name__)
db = get_db()

def create_faiss_index(text_chunks):
    # Generate embeddings for text chunks
    embeddings = model.encode(text_chunks)
    dimension = embeddings.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def search_relevant_chunks(query, index, text_chunks, embeddings, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return relevant_chunks

def get_topic_content(topic_path):
    try:
        with open(topic_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split content into meaningful chunks
        chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        return chunks
    except Exception as e:
        print(f"Error reading topic file: {e}")
        return None

def preprocess_questions(file_path, output_csv):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Splitting the questions based on the pattern **Question X**
    questions = re.split(r'\*\*Question \d+\*\*', content)[1:]
    
    data = []
    
    for q in questions:
        lines = q.strip().split('\n')
        
        question_text = lines[0].strip()
        
        # Removing option characters (e.g., "a.", "b.", "c.", "d.") from the start of options
        options = [re.sub(r'^[a-d]\.\s*', '', line.strip()) for line in lines[1:5]]
        
        correct_option_match = re.search(r'\*\*Correct answer: (.*?)\*\*', q)
        subtopic_match = re.search(r'\*\*Subtopic: (.*?)\*\*', q)
        subtopic_detail_match = re.search(r'\*\*What in that subtopic exactly: (.*?)\*\*', q)
        difficulty_match = re.search(r'\*\*Difficulty: (.*?)\*\*', q)
        
        # Extract and clean the correct option
        correct_option = correct_option_match.group(1) if correct_option_match else ''
        correct_option = re.sub(r'^[a-d]\.\s*', '', correct_option)  # Remove "a.", "b.", etc.
        
        subtopic = subtopic_match.group(1) if subtopic_match else ''
        subtopic_detail = subtopic_detail_match.group(1) if subtopic_detail_match else ''
        difficulty = difficulty_match.group(1) if difficulty_match else ''
        
        data.append([question_text] + options + [correct_option, subtopic, subtopic_detail, difficulty])
    
    df = pd.DataFrame(data, columns=['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Option', 'Subtopic', 'Subtopic Detail', 'Difficulty'])
    
    df.to_csv(output_csv, index=False, encoding='utf-8')
    return df

def generate_questions(topic, subtopic, topic_content=None, model="llama3-70b-8192"):
    # Create context from internal content only
    context = ""
    
    if topic_content:
        context = (
            "INTERNAL COURSE CONTENT:\n" + 
            "\n".join(topic_content) + 
            "\n\n"
        )
    
    prompt = (f"{context}"
             f"Based on the internal content, generate up to 5 multiple choice questions on {topic} under {subtopic}. "
             "Each question should have 4 options (a, b, c, d) with one correct answer. "
             "Go from easy to very hard. Format as follows:\n\n"

             "<question>\n"
             "a.<option1>\n"
             "b.<option2>\n"
             "c.<option3>\n"
             "d.<option4>\n\n"
             "<correct option>\n"
             "<subtopic>\n"
             "<what in that subtopic exactly>\n"
             "<a flag for difficulty (e,m,h)(easy medium hard)>\n"
             "<content reference: quote the specific content this question is based on>\n"
             "<####>"
             "\n\n"
             "for example the structure of output must look exactly as follows permitting no variation:\n" 
             " **Question 1**\n"
             "What is the primary advantage of using Bubble Sort?\n"
             "a. It is the fastest sorting algorithm.\n"
             "b. It is easy to implement.\n"
             "c. It uses the least amount of memory.\n"
             "d. It is used for large datasets.\n\n"
             "**Correct answer: b. It is easy to implement.**\n"
             "**Subtopic: Bubble Sort**\n"
             "**What in that subtopic exactly: Characteristics of Bubble Sort**\n"
             "**Difficulty: e**\n"
             "**Content reference: 'Bubble Sort is one of the simplest sorting algorithms to implement...'**"
    )
     
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7  # Reduced temperature for more focused responses
    )
    return response.choices[0].message.content

def clear_generated_tests():
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir_path = os.path.join(current_dir, "generated_tests")
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def normalize_course_name(course):
    """Normalize course name to match directory structure"""
    course_map = {
        "operating system": "Operating system",
        "operating systems": "Operating system",
        "Operating System": "Operating system",  # Handle the specific case
        "data structures and algorithms": "Data Structures and Algorithms",
        "computer networks": "Computer Networks"
    }
    return course_map.get(course, course)  # Don't lowercase the input to preserve other correct cases

@temp_start_bp.route('/temp_start_bp', methods=['POST'])
def start_test():
    # Clear the generated_tests directory
    clear_generated_tests()

    data = request.json
    email = data.get('email')
    chapter_ = data.get('chapter_')
    
    if not email or not chapter_:
        return jsonify({"error": "Missing email or chapter"}), 400
    
    email = email.strip()
    
    try:
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).get()

        if not query:
            return jsonify({"error": "User not found"}), 404
            
        user_doc = query[0].to_dict()
        course = user_doc.get('course')
        rank = user_doc.get("rank")
        
        if not course:
            return jsonify({"error": "No course found for user"}), 404

        # Normalize the course name
        course = normalize_course_name(course)
        print(f"Normalized course name: {course}")
        
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        content_path = os.path.join(current_dir, '..', 'content', course)
        print(f"Looking for content at: {os.path.abspath(content_path)}")
        
        if not os.path.exists(content_path):
            available_courses = os.listdir(os.path.join(current_dir, '..', 'content'))
            return jsonify({
                "error": f"Course directory not found. Your course: '{course}'. Available courses: {', '.join(available_courses)}"
            }), 404

        questions_text = ""
        for chapter_name in sorted(os.listdir(content_path), key=extract_number):
            chapter_path = os.path.join(content_path, chapter_name)
            
            if os.path.isdir(chapter_path) and extract_number(chapter_name) == int(chapter_):
                topics = sorted(
                    [file for file in os.listdir(chapter_path) 
                     if os.path.isfile(os.path.join(chapter_path, file))],
                    key=extract_number
                )

                for topic in topics:
                    topic_name = get_stripped_name(topic)
                    # Search for relevant content chunks for this topic
                    relevant_chunks = search_relevant_chunks(topic_name, relevant_chunks)
                    questions = generate_questions(chapter_name, topic_name, relevant_chunks)
                    questions_text += questions + "\n"

        # Make sure the output directory exists
        output_dir = os.path.join(current_dir, "generated_tests")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the questions to a file
        filename = f"generated_test_{chapter_}.txt"
        output_file = os.path.join(output_dir, filename)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(questions_text)
        
        return jsonify({
            "success": True, 
            "message": "Test questions generated successfully", 
            "file": filename,
            "rank": rank,
            "chapter": chapter_
        }), 200

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@csv_generation_bp.route('/generate_csv', methods=['POST'])
def generate_csv():
    data = request.json
    chapter_ = data.get('chapter_')
    
    if not chapter_:
        return jsonify({"error": "Missing chapter"}), 400
    
    try:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Ensure we're using the correct absolute paths
        input_file = os.path.join(current_dir, "generated_tests", f"generated_test_{chapter_}.txt")
        output_file = os.path.join(current_dir, "generated_tests", f"output_{chapter_}.csv")
        
        # Check if the text file exists
        if not os.path.exists(input_file):
            return jsonify({"error": f"Test file not found: {input_file}"}), 404
        
        # Process the file to create the CSV
        df = preprocess_questions(input_file, output_file)
        
        return jsonify({
            "success": True, 
            "message": "CSV generated successfully",
            "csv_file": output_file,
            "question_count": len(df)
        }), 200
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500