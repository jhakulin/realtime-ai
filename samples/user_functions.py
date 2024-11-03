import json
import datetime
from typing import Callable, Any, Set
import openai
from PIL import Image
import mss
import io
import os
import base64
from pathlib import Path

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


def take_screenshot_and_analyze() -> str:
    """
    Captures a screenshot, sends it to the specified OpenAI vision model for analysis,
    and returns the analysis result.

    :return: The analysis result as a JSON string.
    :rtype: str
    """

    try:
        openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None
    
    # Step 1: Capture the screenshot
    print("Capturing screenshot...")
    with mss.mss() as sct:
        monitor = sct.monitors[0]  # 0 is the first monitor; adjust if multiple monitors are used
        screenshot = sct.grab(monitor)
        img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
    
    # Optional: Save the screenshot locally (uncomment if needed)
    # img.save('screenshot.png')

    # Step 2: Convert the image to binary data
    print("Converting image to binary data...")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_bytes = img_byte_arr.read()

    # Step 3: Encode the image in base64
    print("Encoding image in base64...")
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Step 4: Prepare the payload for OpenAI API
    print("Sending image to OpenAI for analysis...")
    content = []
    content.append({"type": "text", "text": "Please analyze the following image and provide a detailed description."})
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}", "detail": "high"}}]
    messages = [{"role": "user", "content": content}]

    try:
        # Step 5: Call OpenAI's ChatCompletion API with the image
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5,
            max_tokens=2000
        )
        
        # Step 6: Extract the analysis result from the response
        analysis = response.choices[0].message.content
        print(f"Analysis: {analysis}")
        return json.dumps({"analysis": analysis})

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def take_screenshot_and_save() -> str:
    """
    Captures a screenshot and saves it to the local filesystem.

    :return: The path to the saved screenshot.
    :rtype: str
    """
    print("Capturing screenshot...")
    with mss.mss() as sct:
        monitor = sct.monitors[0]  # 0 is the first monitor; adjust if multiple monitors are used
        screenshot = sct.grab(monitor)
        img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
    
    # Save the screenshot locally
    img.save('screenshot.png')

    # open the saved image in the default image viewer
    img.show()

    print(f"Screenshot saved: {Path.cwd() / 'screenshot.png'}")
    return json.dumps({"path": "screenshot.png"})


# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
    fetch_current_datetime,
    fetch_weather,
    send_email,
    take_screenshot_and_analyze,
    take_screenshot_and_save
}
