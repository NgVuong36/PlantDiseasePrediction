import os
from google import genai
from dotenv import load_dotenv

def GeminiChat(result) -> str:
    """
    Gọi API Gemini
    :param result: string
    :return: string
    """
    load_dotenv()

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=f"Bạn là một chuyên gia nông nghiệp, hãy mô tả tình trạng cây trồng của tôi: {result}. Hãy loại bỏ các ký tự đặc biệt, trả lời ngắn gọn và không cần đề mục.",
    )
    return response.text

