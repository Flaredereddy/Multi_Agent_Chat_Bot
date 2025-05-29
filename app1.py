import os
import subprocess
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import requests
import re
import csv
import json
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize Flask app
app = Flask(__name__)

# 🔧 Set up paths
BASE_PATH = "/Users/dereddylikhith/Desktop/Combined_Chatbot"
CONTEXT_FOLDER = os.path.join(BASE_PATH, "context")
REVIEW_FOLDER = os.path.join(BASE_PATH, "reviews")
REVIEW_CSV_PATH = os.path.join(REVIEW_FOLDER, "reviews_with_gpt4o_sentiment.csv")
VECTOR_DB_DIR = os.path.join(BASE_PATH, "vectorstore")
YOLOV5_DIR = os.path.join(BASE_PATH, "yolov5")
WEIGHTS_PATH = os.path.join(BASE_PATH, "weights/best.pt")
UPLOADS_DIR = os.path.join(BASE_PATH, "static/uploads")

# Ensure directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Set API keys
os.environ["OPENAI_API_KEY"] = "TCs_tFwh1g8v8hbWvdfvdffddgHmIl-Kx3CaFbtM0uwUCEA"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ldfsdfv8e018vdffsdv892e4"
os.environ["LANGCHAIN_PROJECT"] = "CORA Chatbot"

# Feedback logging
FEEDBACK_LOG = []
FEEDBACK_FILE = "/tmp/feedback_log.csv"
last_question = ""
last_response = ""

# State-to-city mapping for weather queries
STATE_TO_CITY = {
    "alabama": "Birmingham", "alaska": "Anchorage", "arizona": "Phoenix", "arkansas": "Little Rock",
    "california": "Los Angeles", "colorado": "Denver", "connecticut": "Hartford", "delaware": "Wilmington",
    "florida": "Miami", "georgia": "Atlanta", "hawaii": "Honolulu", "idaho": "Boise", "illinois": "Chicago",
    "indiana": "Indianapolis", "iowa": "Des Moines", "kansas": "Wichita", "kentucky": "Louisville",
    "louisiana": "New Orleans", "maine": "Portland", "maryland": "Baltimore", "massachusetts": "Boston",
    "michigan": "Detroit", "minnesota": "Minneapolis", "mississippi": "Jackson", "missouri": "St. Louis",
    "montana": "Billings", "nebraska": "Omaha", "nevada": "Las Vegas", "new hampshire": "Manchester",
    "new jersey": "Newark", "new mexico": "Albuquerque", "new york": "New York", "north carolina": "Charlotte",
    "north dakota": "Fargo", "ohio": "Columbus", "oklahoma": "Oklahoma City", "oregon": "Portland",
    "pennsylvania": "Philadelphia", "rhode island": "Providence", "south carolina": "Charleston",
    "south dakota": "Sioux Falls", "tennessee": "Nashville", "texas": "Houston", "utah": "Salt Lake City",
    "vermont": "Burlington", "virginia": "Richmond", "washington": "Seattle", "west virginia": "Charleston",
    "wisconsin": "Milwaukee", "wyoming": "Cheyenne"
}

# Load the existing vectorstore
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding_model)

# Define Agents
class RAGAgent:
    def __init__(self, vectorstore):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.context_retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 10, "filter": {"type": {"$in": ["docx_text", "image_ocr"]}}}
        )
        prompt_template = """
        You are an expert on toilet specifications. Use the following pieces of context to answer the question as accurately as possible. The context may include technical specifications (e.g., dimensions, flush rates, features) extracted from documents or images, as well as user reviews.

        - If the question asks for a list of all toilet models or the total number of toilet models, ensure you extract all model names mentioned in the context (e.g., Aluvia_Plus, Aluvia, NYREN, Cascade, Smart, Cima) and list them with their specifications if available.
        - If the question involves flush rates or water usage, interpret "best" as the lowest flush rate for water efficiency (e.g., 3.8 LPF) unless specified otherwise.
        - Extract and include any relevant details such as dimensions, flush rates, ADA compliance, flush types, or user feedback.
        - If you cannot find the answer in the context, say so, but do not guess or make up information.

        Question: {question}

        Context:
        {context}

        Answer:
        """
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.context_retriever, return_source_documents=True, chain_type_kwargs={"prompt": self.prompt}
        )

    def query(self, question):
        if any(keyword in question.lower() for keyword in [
            "different kinds of toilet models", "list all toilet models", "what toilet models are available",
            "all models of toilet", "list of toilet models", "available toilet models",
            "what are the toilet models", "show all toilet models", "different toilets",
            "types of toilets", "kinds of toilets", "models of toilets"
        ]):
            model_names = self.extract_all_models()
            result = self.qa_chain.invoke({"query": f"Provide specifications for the following toilet models: {', '.join(model_names)}"})
            return result["result"]
        if "how many models" in question.lower() and "toilet" in question.lower:
            model_names = self.extract_all_models()
            num_models = len(model_names)
            result = self.qa_chain.invoke({"query": f"There are {num_models} toilet models: {', '.join(model_names)}. Provide their specifications."})
            return result["result"]
        if any(keyword in question.lower() for keyword in ["best for old people", "suitable for elderly", "good for seniors", "toilet for old", "elderly friendly"]):
            result = self.qa_chain.invoke({"query": question})
            return result["result"]
        result = self.qa_chain.invoke({"query": question})
        return result["result"]

    def extract_all_models(self):
        docs = vectorstore.get()["metadatas"]
        model_names = set()
        for doc in docs:
            source = doc.get("source", "")
            if source and source.lower().endswith(".docx"):
                model = source.replace("Sanitario_", "").replace(".docx", "")
                if model:
                    model_names.add(model)
        return sorted(list(model_names))

class WeatherAgent:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(self, location):
        api_key = "c4sfsfdffvc4"  # OpenWeatherMap API key provided by user
        if location.lower() in STATE_TO_CITY:
            city = STATE_TO_CITY[location.lower()]
        else:
            city = location
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        try:
            response = requests.get(url)
            data = response.json()
            print(f"Weather API Response for {city}: {data}")
            if data["cod"] == 200:
                weather_desc = data["weather"][0]["description"]
                temp = data["main"]["temp"]
                return f"The weather in {city} is {weather_desc} with a temperature of {temp}°C."
            else:
                return f"Could not fetch weather data: {data.get('message', 'Unknown error')}"
        except Exception as e:
            return f"Error fetching weather: {str(e)}"

class WebSearchAgent:
    # Alpha Vantage API key for S&P 500 price queries
    ALPHA_VANTAGE_API_KEY = "LdfdsfvddscvEE"

    # Brave Search API key for web search
    BRAVE_API_KEY = "BSdvsdvdsdfred3r23e32fdcsdsfsdc327o"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_sp500_price(self):
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=^GSPC&apikey={self.ALPHA_VANTAGE_API_KEY}"
        try:
            response = requests.get(url)
            data = response.json()
            print(f"Alpha Vantage API Response for S&P 500:\n{json.dumps(data, indent=2)}")
            if "Global Quote" in data and "05. price" in data["Global Quote"]:
                price = float(data["Global Quote"]["05. price"])
                return price
            else:
                return None
        except Exception as e:
            print(f"Exception in get_sp500_price: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def brave_search(self, query):
        url = f"https://api.search.brave.com/res/v1/web/search?q={query.replace(' ', '+')}"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.BRAVE_API_KEY
        }
        try:
            response = requests.get(url, headers=headers)
            data = response.json()
            print(f"Brave Search API Response for '{query}':\n{json.dumps(data, indent=2)}")
            if "web" in data and data["web"]["results"]:
                # Scan the top 3 results for a description containing a numerical price
                for i, result in enumerate(data["web"]["results"][:3]):  # Check top 3 results
                    description = result.get("description", "No information found.")
                    source_title = result.get("title", "Unknown Source")
                    # Use regex to find a numerical value (e.g., "5,800", "$5,800", "5800", "5,800 points")
                    price_match = re.search(r'[\$\s]?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:USD|points)?', description, re.IGNORECASE)
                    if price_match:
                        print(f"Extracted Description from result {i+1}: {description}")
                        formatted_result = f"Source: {source_title}\nResult: {description}"
                        return formatted_result
                # If no price is found, return the first result with a fallback message
                description = data["web"]["results"][0].get("description", "No information found.")
                source_title = data["web"]["results"][0].get("title", "Unknown Source")
                print(f"No price found in top 3 results. Using first result: {description}")
                formatted_result = f"Source: {source_title}\nResult: {description}\nNote: I couldn't find the exact price. Please check a financial news source like CNBC or Yahoo Finance for the latest price."
                return formatted_result
            else:
                print("No web results found in the API response.")
                return "Sorry, I couldn't find the information on the web. The Brave Search API returned no results."
        except Exception as e:
            print(f"Exception in brave_search: {str(e)}")
            return f"Error performing web search: {str(e)}"

    def query(self, query):
        query_lower = query.lower()
        if "price of" in query_lower and "sp500" in query_lower:
            print("Fetching S&P 500 price via Alpha Vantage API")
            price = self.get_sp500_price()
            if price is not None:
                return f"The current S&P 500 price is {price:.2f} points as of the latest data (Source: Alpha Vantage API)."
            else:
                return "Unable to fetch the S&P 500 price at this time. Trying a web search instead."
        # Fallback to Brave Search for general queries or if S&P 500 fetch fails
        print(f"Performing web search for query: {query}")
        return self.brave_search(query)

class YOLOv5Agent:
    def query(self, image_path):
        if not image_path:
            return {"text": "No image provided for detection.", "image_url": None, "detected_models": []}
        
        output_dir = os.path.join(YOLOV5_DIR, "runs/detect")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        command = f"python {os.path.join(YOLOV5_DIR, 'detect.py')} --weights {WEIGHTS_PATH} --img 640 --conf 0.25 --source {image_path} --device cpu --project {output_dir} --name exp_{timestamp} --exist-ok"
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            exp_path = os.path.join(output_dir, f"exp_{timestamp}")
            processed_images = [f for f in os.listdir(exp_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if processed_images:
                processed_image_path = os.path.join(exp_path, processed_images[0])
                processed_filename = f"processed_{timestamp}.jpg"
                processed_image_new_path = os.path.join(UPLOADS_DIR, processed_filename)
                os.rename(processed_image_path, processed_image_new_path)
                image_url = f"/static/uploads/{processed_filename}"
                detections = []
                for line in result.stdout.splitlines():
                    if line.startswith("image"):
                        parts = line.split(":")
                        if len(parts) > 1:
                            detection_info = parts[1].split("Done")[0].strip()
                            detections.append(detection_info)
                # Extract model names from detections
                detected_models = []
                for detection in detections:
                    for model in ["Aquapro", "Montecarlo", "Smart"]:
                        if model.lower() in detection.lower():
                            detected_models.append(model)
                            break
                return {"text": "Detected toilet models:\n" + "\n".join(detections), "image_url": image_url, "detected_models": detected_models}
            else:
                return {"text": "No detections found.", "image_url": None, "detected_models": []}
        except Exception as e:
            return {"text": f"Error running YOLOv5: {str(e)}", "image_url": None, "detected_models": []}

# Dispatcher for multi-agent system
class Dispatcher:
    def __init__(self):
        self.rag_agent = RAGAgent(vectorstore)
        self.weather_agent = WeatherAgent()
        self.websearch_agent = WebSearchAgent()
        self.yolov5_agent = YOLOv5Agent()

    def route(self, message, image_path=None):
        message_lower = message.lower() if message else ""
        print(f"Received query - Message: '{message}', Image Path: {image_path}")

        # Greeting detection
        if message and bool(re.fullmatch(r"(hi|hello|hey|hola|start|begin)", message.strip().lower())):
            print("Routing to greeting response")
            return {
                "text": (
                    "👋 Hello! I’m **CORA**, your Corona Toilet Assistant.\n\n"
                    "I can help with:\n"
                    "- 🚽 Toilet specifications\n"
                    "- 💧 Water usage\n"
                    "- 🧾 Customer reviews\n"
                    "- ♿ ADA compliance\n"
                    "- 🌦️ Weather updates\n"
                    "- 🌐 Web searches (e.g., 'price of sp500')\n"
                    "- 🖼️ Identify toilet models in images (upload an image or type 'Identify this toilet model')\n\n"
                    "You can send a message, upload an image, or both!\n\n"
                    "How can I assist you today?"
                ),
                "image_url": None
            }

        # Image detection (if image is provided or text indicates image-related query)
        image_related_keywords = ["identify", "detect", "what is this", "toilet model", "image"]
        if image_path or (message and any(keyword in message_lower for keyword in image_related_keywords)):
            print("Routing image query to YOLOv5 Agent")
            yolov5_result = self.yolov5_agent.query(image_path)
            detected_models = yolov5_result["detected_models"]
            print(f"YOLOv5 Agent result: {yolov5_result}")
            
            # If a text query is provided and detections are found, pass the detected models to RAG Agent
            if message and detected_models:
                rag_query = f"{message}\nDetected models in the image: {', '.join(detected_models)}."
                print(f"Routing combined query to RAG Agent: {rag_query}")
                try:
                    rag_response = self.rag_agent.query(rag_query)
                    print(f"RAG Agent response for combined query: {rag_response}")
                    combined_text = f"{yolov5_result['text']}\n\nAdditional Information:\n{rag_response}"
                    return {"text": combined_text, "image_url": yolov5_result["image_url"]}
                except Exception as e:
                    print(f"RAG Agent error for combined query: {str(e)}")
                    return {"text": f"Error processing combined query: {str(e)}", "image_url": yolov5_result["image_url"]}
            elif detected_models:
                # Image-only case: Provide detection results and basic specs
                rag_query = f"Provide specifications for the following toilet models: {', '.join(detected_models)}"
                print(f"Routing image-only query to RAG Agent: {rag_query}")
                try:
                    rag_response = self.rag_agent.query(rag_query)
                    print(f"RAG Agent response for image-only query: {rag_response}")
                    combined_text = f"{yolov5_result['text']}\n\nSpecifications:\n{rag_response}"
                    return {"text": combined_text, "image_url": yolov5_result["image_url"]}
                except Exception as e:
                    print(f"RAG Agent error for image-only query: {str(e)}")
                    return {"text": f"Error processing image-only query: {str(e)}", "image_url": yolov5_result["image_url"]}
            else:
                # No detections or no further query
                return yolov5_result

        # Weather query
        if message and "weather" in message_lower:
            print("Routing to Weather Agent")
            for state in STATE_TO_CITY.keys():
                if state in message_lower:
                    response = self.weather_agent.query(state)
                    print(f"Weather Agent response: {response}")
                    return {"text": response, "image_url": None}
            words = message_lower.split()
            for word in words:
                if word != "weather" and word != "in":
                    response = self.weather_agent.query(word)
                    print(f"Weather Agent response: {response}")
                    return {"text": response, "image_url": None}
            return {"text": "Please specify a location for the weather query.", "image_url": None}

        # Web search query
        if message and any(keyword in message_lower for keyword in ["tell me about", "search for", "what is", "price of"]):
            print("Routing to WebSearch Agent")
            response = self.websearch_agent.query(message)
            print(f"WebSearch Agent response: {response}")
            return {"text": response, "image_url": None}

        # RAG query (default for text-only)
        if message:
            print(f"Routing text query to RAG Agent: {message}")
            try:
                rag_response = self.rag_agent.query(message)
                print(f"RAG Agent response: {rag_response}")
                if rag_response.strip().lower() in ["i don't know", "no information found", "none"]:
                    rag_response = (
                        "🤔 I couldn’t find a specific answer. Try asking something like:\n"
                        "- 'List toilets under 4 Lpf'\n"
                        "- 'Show ADA-compliant models'\n"
                        "- 'Compare Smart vs Cascade toilets'\n"
                        "- 'Weather in New Jersey'\n"
                        "- 'What is the price of the sp500?'\n"
                        "- Or upload an image to identify a toilet model."
                    )
                return {"text": rag_response, "image_url": None}
            except Exception as e:
                print(f"RAG Agent error: {str(e)}")
                return {"text": f"Error processing query: {str(e)}", "image_url": None}

        return {"text": "Please provide a message or upload an image.", "image_url": None}

# Chatbot logic
dispatcher = Dispatcher()

def multi_agent_chatbot(message, image_path=None):
    global last_question, last_response
    result = dispatcher.route(message, image_path)
    last_question = message if message else "Image upload"
    last_response = result["text"]
    return result

# Log feedback
def log_feedback(question, answer, feedback):
    global FEEDBACK_LOG
    timestamp = datetime.now().isoformat(timespec='seconds')
    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "feedback": feedback
    }
    FEEDBACK_LOG.append(log_entry)
    try:
        file_exists = os.path.exists(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["timestamp", "question", "answer", "feedback"])
            writer.writerow([timestamp, question, answer, feedback])
    except Exception as e:
        print(f"Failed to write feedback to file: {str(e)}")
    return f"Feedback received: **{feedback.replace('_', ' ').title()}**"

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form.get('message', '')
    image = request.files.get('image', None)
    image_path = None

    # Save the uploaded image if present
    if image and image.filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(UPLOADS_DIR, f"uploaded_{timestamp}.jpg")
        image.save(image_path)

    # Process the request
    result = multi_agent_chatbot(message, image_path)
    return jsonify({'response': result["text"], 'image_url': result["image_url"]})

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_value = request.form['feedback']
    label = "thumbs_up" if feedback_value == "👍" else "thumbs_down"
    response = log_feedback(last_question, last_response, label)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)