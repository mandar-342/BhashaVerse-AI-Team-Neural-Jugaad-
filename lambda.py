import json
import boto3
import base64

# ─── AWS CLIENTS ───────────────────────────────────────────
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
polly   = boto3.client("polly",           region_name="ap-south-1")

# ─── CORS HEADERS ──────────────────────────────────────────
CORS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

# ─── SYSTEM PROMPT ─────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful farming assistant for rural India.
Answer the farmer's question DIRECTLY and SPECIFICALLY.
Do NOT introduce yourself. Do NOT say what you can do. Just answer the question asked.
Keep answer to 3-5 sentences. Be practical with specific medicine names, doses, helpline numbers.
Match the language of the question — Hindi = Hindi answer, Marathi = Marathi, English = English.

Help with these topics ONLY:
1. FARMING: crop diseases, fertilizers, irrigation, pest control, sowing schedules
2. GOVERNMENT SCHEMES: PM-KISAN (6000/year), PM Fasal Bima, Ayushman Bharat, KCC
3. MARKET PRICES: mandi rates, MSP, selling tips
4. HEALTH: basic health advice, govt health services, emergency numbers
5. WEATHER: farming-relevant weather advice

RULES:
- Answer in the SAME LANGUAGE as the question
- Keep answers SHORT: 3-5 sentences maximum
- Always give PRACTICAL, ACTIONABLE advice
- Include specific helpline numbers when relevant
- Use simple words a farmer can understand
- If outside scope, redirect to Kisan Helpline 1551

Key helpline numbers:
- Kisan Helpline: 1551 (farming, 24x7, free)
- Ayushman Bharat: 14555
- PM-KISAN: 155261
- Fasal Bima: 14447
- Emergency: 108"""

# ─── LANGUAGE CONFIG ───────────────────────────────────────
LANG_CONFIG = {
    "hi": {"polly_voice": "Aditi",   "polly_lang": "hi-IN"},
    "mr": {"polly_voice": "Aditi",   "polly_lang": "hi-IN"},
    "ta": {"polly_voice": "Aditi",   "polly_lang": "hi-IN"},
    "te": {"polly_voice": "Aditi",   "polly_lang": "hi-IN"},
    "bn": {"polly_voice": "Aditi",   "polly_lang": "hi-IN"},
    "en": {"polly_voice": "Raveena", "polly_lang": "en-IN"},
}


def lambda_handler(event, context):
    # CORS preflight
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS, "body": ""}

    try:
        body     = json.loads(event.get("body", "{}"))
        query    = (body.get("query") or body.get("message") or "").strip()
        language = (body.get("language") or "hi").strip().lower()
        want_tts = True

        if not query:
            return _error(400, "Missing 'query' in request body")

        if language not in LANG_CONFIG:
            language = "hi"

        # Call Bedrock Nova Lite
        response = bedrock.invoke_model(
            modelId="apac.amazon.nova-lite-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": [{"text": query}]}],
                "system":   [{"text": SYSTEM_PROMPT}],
                "inferenceConfig": {
                    "maxTokens":   300,
                    "temperature": 0.4,
                    "topP":        0.85,
                },
            }),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        answer = result["output"]["message"]["content"][0]["text"].strip()

        # Optional Polly TTS
        audio_b64 = None
        if want_tts and len(answer) < 600:
            try:
                lc = LANG_CONFIG[language]
                polly_resp = polly.synthesize_speech(
                    Text=answer[:500],
                    OutputFormat="mp3",
                    VoiceId=lc["polly_voice"],
                    LanguageCode=lc["polly_lang"],
                    Engine="standard",
                )
                audio_b64 = base64.b64encode(
                    polly_resp["AudioStream"].read()
                ).decode("utf-8")
            except Exception as pe:
                print(f"Polly error (non-fatal): {pe}")

        resp = {
            "answer":   answer,
            "language": language,
            "source":   "amazon-bedrock-nova-lite",
        }
        if audio_b64:
            resp["audio_b64"] = audio_b64

        return {
            "statusCode": 200,
            "headers": {**CORS, "Content-Type": "application/json"},
            "body": json.dumps(resp, ensure_ascii=False),
        }

    except Exception as e:
        print(f"Lambda error: {e}")
        return _error(500, str(e))


def _error(code, msg):
    return {
        "statusCode": code,
        "headers": {**CORS, "Content-Type": "application/json"},
        "body": json.dumps({"error": msg}),
    }