"""
Vision Detector — Groq Llama 4 Scout (Free, 14400 req/day)
Catches: colorized historical photos, AI-generated images, impossible scenarios
"""
import os, logging, json, base64

logger = logging.getLogger(__name__)

PROMPT = """Analyze this image. Reply ONLY with JSON: {"fake_probability": <0.0-1.0>, "reason": "<one sentence>"}

Rules:
- 0.9+ if a modern famous person (born after 1940) appears in a pre-1970s photo setting
- 0.85+ if clearly AI-generated (Midjourney/DALL-E/Stable Diffusion — too perfect, painterly)
- 0.8+ if a deepfake or obvious digital manipulation
- 0.75+ if AI-colorized old black & white photo
- 0.05 or less for normal real photographs of real people

No other text, just the JSON."""


def analyze_with_claude(image_bytes: bytes):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        mime = "image/png" if image_bytes[:4] == b'\x89PNG' else "image/jpeg"
        b64 = base64.standard_b64encode(image_bytes).decode()

        resp = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": PROMPT}
            ]}],
            max_tokens=80,
            temperature=0.1,
        )

        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1:
            return None

        result = json.loads(raw[start:end])
        prob = max(0.0, min(1.0, float(result.get("fake_probability", 0.0))))
        reason = result.get("reason", "")
        logger.info(f"Llama4 vision: prob={prob:.2f} reason='{reason}'")
        return {"fake_probability": round(prob, 4), "reason": reason}

    except Exception as e:
        logger.debug(f"Vision AI skipped: {e}")
        return None
