import json
import datetime
from typing import Callable, Any, Set
import openai
from pathlib import Path
import io, os, base64


# These are the user-defined functions that can be called by the agent.

def fetch_current_datetime() -> str:
    """
    Get the current time as a JSON string.

    :return: The current time in JSON format.
    :rtype: str
    """
    current_time = datetime.datetime.now()
    time_json = json.dumps({"current_time": current_time.strftime("%Y-%m-%d %H:%M:%S")})
    return time_json


def fetch_weather(location: str) -> str:
    """
    Fetches the weather information for the specified location.

    :param location (str): The location to fetch weather for.
    :return: Weather information as a JSON string.
    :rtype: str
    """
    # In a real-world scenario, you'd integrate with a weather API.
    # Here, we'll mock the response.
    mock_weather_data = {"New York": "Sunny, 25°C", "London": "Cloudy, 18°C", "Tokyo": "Rainy, 22°C"}
    weather = mock_weather_data.get(location, "Weather data not available for this location.")
    weather_json = json.dumps({"weather": weather})
    return weather_json


def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Sends an email with the specified subject and body to the recipient.

    :param recipient (str): Email address of the recipient.
    :param subject (str): Subject of the email.
    :param body (str): Body content of the email.
    :return: Confirmation message.
    :rtype: str
    """
    # In a real-world scenario, you'd use an SMTP server or an email service API.
    # Here, we'll mock the email sending.
    print(f"Sending email to {recipient}...")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")

    message_json = json.dumps({"message": f"Email successfully sent to {recipient}."})
    return message_json


def _generate_chat_completion(ai_client, model, messages):
    print(f"generate_chat_completion, messages: {messages}")
    print(f"generate_chat_completion, model: {model}")

    try:
        # Generate the chat completion
        response = ai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        print(f"generate_chat_completion, response: {response}")

        # Extract the content of the first choice
        if response.choices and response.choices[0].message:
            message_content = response.choices[0].message.content
        else:
            message_content = "No response"

        return json.dumps({"result": message_content})
    except Exception as e:
        error_message = f"Failed to generate chat completion: {str(e)}"
        print(error_message)
        return json.dumps({"function_error": error_message})


def _screenshot_to_bytes() -> bytes:
    """
    Captures a screenshot and returns it as binary data.

    :return: The screenshot as binary data.
    :rtype: bytes
    """
    from PIL import Image
    import mss

    with mss.mss() as sct:
        monitor = sct.monitors[0]  # 0 is the first monitor; adjust if multiple monitors are used
        screenshot = sct.grab(monitor)
        img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
    
    # Convert the image to binary data
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_bytes = img_byte_arr.read()
    return img_bytes


def _analyze_image(img_base64: str, system_input: str, user_input: str, filename: str) -> str:
    """
    Analyzes the given image and returns the analysis result.

    :param img_base64 (str): Base64 encoded image data.
    :param system_input (str): System input for the analysis.
    :param user_input (str): User input for the analysis.
    :return: The analysis result.
    :rtype: str
    """

    try:
        openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_input
                }
            ],
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": user_input
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": "high"
                    }
                },
            ],
        }
    ]

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5,
            max_tokens=2000
        )
        
        # Extract the analysis result from the response
        analysis = response.choices[0].message.content
        print(f"User input: {user_input}")
        print(f"Analysis: {analysis}")

        # Create user message using the user input and the analysis result
        #user_message = f"User input: {user_input}\nAnalysis: {analysis}"
        #o1_messages = [{"role": "user", "content": user_message}]
        #o1_response = _generate_chat_completion(openai_client, "o1-mini", o1_messages)
        #print(f"O1 response: {o1_response}")

        # Show the analysis result in code
        with open(filename, "w") as f:
            f.write(analysis)
        os.system(f"code {filename}")

        return json.dumps({"analysis": analysis})
    
    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        return json.dumps({"function_error": error_message})


def review_highlighted_code() -> str:
    """
    Captures a screenshot, sends it to the specified OpenAI model for analysis,
    and returns the analysis result.

    :return: The analysis result as a JSON string.
    :rtype: str
    """    
    # Capture a screenshot and convert it to base64
    img_bytes = _screenshot_to_bytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return _analyze_image(img_base64=img_base64, 
                          system_input="You are expert in analyzing images to text. If the image contains highlighted part, focus on that.", 
                          user_input="Review the highlighted code and provide detailed feedback.",
                          filename="highlighted_code_analysis.md")


def translate_highlighted_text(language: str) -> str:
    """
    Captures a screenshot, sends it to the specified OpenAI model for analysis,
    and returns the analysis result.

    :return: The analysis result as a JSON string.
    :rtype: str
    """    
    # Capture a screenshot and convert it to base64
    img_bytes = _screenshot_to_bytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return _analyze_image(img_base64=img_base64, 
                          system_input=f"You are expert in translating text to different languages.", 
                          user_input=f"Translate the highlighted text to {language}.",
                          filename="highlighted_text_translation.md")


def explain_highlighted_text() -> str:
    """
    Captures a screenshot, sends it to the specified OpenAI model for analysis,
    and returns the analysis result.

    :return: The analysis result as a JSON string.
    :rtype: str
    """    
    # Capture a screenshot and convert it to base64
    img_bytes = _screenshot_to_bytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return _analyze_image(img_base64=img_base64, 
                          system_input="You are expert in explaining text. If the image contains highlighted text, provide the explanation of that.", 
                          user_input="Explain the highlighted text in detail, in understandable language.",
                          filename="highlighted_text_explanation.md")


def take_screenshot_and_analyze(user_input: str) -> str:
    """
    Captures a screenshot, sends it to the specified OpenAI model for analysis,
    and returns the analysis result.

    :param user_input (str): User input request as it was given by user for screenshot analysis and actions.
    
    :return: The analysis result as a JSON string.
    :rtype: str
    """    
    # Capture a screenshot and convert it to base64
    img_bytes = _screenshot_to_bytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return _analyze_image(img_base64=img_base64, 
                          system_input="Analyze the screenshot and provide all details from it. If the image contains e.g. code or highlighted parts, provide the exact analysis of that.", 
                          user_input=user_input,
                          filename="screenshot_analysis.md")


def take_screenshot_and_show() -> str:
    """
    Captures a screenshot and displays it to the user.

    :return: The path to the saved screenshot.
    :rtype: str
    """
    from PIL import Image
    import mss

    print("Capturing screenshot...")
    with mss.mss() as sct:
        monitor = sct.monitors[0]  # 0 is the first monitor; adjust if multiple monitors are used
        screenshot = sct.grab(monitor)
        img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
    
    # open the saved image in the default image viewer
    img.show()

    return json.dumps({"result": "Screenshot captured and displayed."})


# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
    fetch_current_datetime,
    fetch_weather,
    send_email,
    take_screenshot_and_analyze,
    take_screenshot_and_show,
    review_highlighted_code,
    translate_highlighted_text,
    explain_highlighted_text
}
