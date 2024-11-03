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


def take_screenshot_and_analyze(user_input: str) -> str:
    """
    Captures a screenshot, sends it to the specified OpenAI model for analysis,
    and returns the analysis result.

    :param user_input (str): User input request as it was given by user for screenshot analysis and actions.
    
    :return: The analysis result as a JSON string.
    :rtype: str
    """
    from PIL import Image
    import mss

    try:
        openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None
    
    # Capture a screenshot
    with mss.mss() as sct:
        monitor = sct.monitors[0]  # 0 is the first monitor; adjust if multiple monitors are used
        screenshot = sct.grab(monitor)
        img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
    
    # Convert the image to binary data
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_bytes = img_byte_arr.read()

    # Encode the image in base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Prepare the payload for OpenAI API
    content = []
    content.append({"role": "system", "content": "Provide an answer that focuses solely on the information requested, avoiding personal references or perspectives. Ensure the response is objective and directly addresses the question or topic"})
    content.append({"type": "text", "text": user_input})
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}", "detail": "high"}}]
    messages = [{"role": "user", "content": content}]

    try:
        # Call OpenAI's ChatCompletion API with the image
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

        # Show the analysis result in code
        with open("analysis.md", "w") as f:
            f.write(analysis)
        os.system("code analysis.md")

        return json.dumps({"analysis": analysis})

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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
    take_screenshot_and_show
}
