from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from typing import Dict
import time
import requests
import json
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from config import Config

OPENAI_API_KEY = Config.read('app', 'gpt.key')
CLOVAX_API_KEY = Config.read('app', 'clova.key')
CLOVAX_PRIMARY_KEY = Config.read('app', 'clova.primary.key')
GEMINI_API_KEY = Config.read('app', 'gemini.key')
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

# ChatGPT 엔드포인트
@app.post("/chatgpt")
async def chatgpt_api(request: Request):
    data = await request.json()
    print("ChatGPT 시작 - ", data)
    prompt = data.get('prompt')
    keyword = data.get('keyword')
    YOUR_API_KEY = OPENAI_API_KEY

    start = time.time()
    client = OpenAI(api_key=YOUR_API_KEY)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": keyword
            }
        ]
    )

    end = time.time()
    sec = (end - start)
    response_text = completion.choices[0].message.content if completion.choices else None

    # 응답이 비어있으면 예외 처리
    if not response_text:
        raise HTTPException(status_code=500, detail="ChatGPT returned no content")

    return {"response": response_text, "time": round(sec, 4)}


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        with requests.post(self._host + '/testapp/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
            process_next_line = False
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    # 'event:result' 이벤트를 감지하면, 다음 라인을 처리하기 위해 플래그 설정
                    if 'event:result' in decoded_line:
                        process_next_line = True
                        continue  # 현재 라인은 무시하고 다음 라인으로 넘어감

                    # 이전 라인에서 'event:result' 이벤트가 감지되었다면, 이 라인을 처리
                    if process_next_line:
                        result = decoded_line
                        if result.startswith('data:'):
                            result = result[5:]
                        data = json.loads(result)
                        text = data["message"]["content"]
                        counttokens = int(data["inputLength"]) + int(data["outputLength"])
                        print("#" * 50)
                        print("ClovaX: ", text)
                        return text, counttokens

# ClovaX 엔드포인트
@app.post("/clovax")
async def clovax_api(request: Request):
    data = await request.json()
    print("ClovaX 시작 - ", data)
    prompt = data.get('prompt')
    keyword = data.get('keyword')

    start = time.time()

    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=CLOVAX_API_KEY,
        api_key_primary_val=CLOVAX_PRIMARY_KEY,
        request_id='ea4e60f460b94843b1c02563bc3277f1'
    )

    preset_text = [{"role":"system","content":prompt},
                   {"role":"user","content":keyword}]

    request_data = {
        'messages': preset_text,
        'topP': 0.9,
        'topK': 0,
        'maxTokens': 256,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': ['###', '###'],
        'includeAiFilters': True,
        'seed': 0
    }

    end = time.time()
    sec = (end - start)
    contents, tokken = completion_executor.execute(request_data)

    if not contents:
        raise HTTPException(status_code=500, detail="ClovaX returned no content")

    return {"response": contents, "tokens": tokken, "time": round(sec, 4)}

# Gemini 엔드포인트
@app.post("/gemini")
async def gemini_api(request: Request):
    data = await request.json()
    print("Gemini 시작 - ", data)
    prompt = data.get('prompt')
    keyword = data.get('keyword')
    YOUR_API_KEY = GEMINI_API_KEY

    start = time.time()

    genai.configure(api_key=YOUR_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=prompt)

    user_prompt = keyword
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            temperature=1.0)
    )
    end = time.time()
    sec = (end - start)

    if not response or not response.text:
        raise HTTPException(status_code=500, detail="Gemini returned no content")

    return {"response": response.text, "time": round(sec, 4)}

# 우선순위 로직: ChatGPT -> ClovaX -> Gemini
@app.post("/fallback")
async def fallback_api(request: Request):
    # 요청 데이터 재추출 (같은 request를 재사용하기 위함)
    data = await request.json()
    print(data)
    prompt = data.get('prompt')
    keyword = data.get('keyword')

    # 먼저 ChatGPT 호출 시도
    try:
        # ChatGPT 호출
        # 새로운 request 생성 없이 동일한 데이터 사용을 위해 임시로 data를 다시 전달하는 예시
        chatgpt_response = await chatgpt_api(request)
        return chatgpt_response
    except Exception as e:
        print("ChatGPT call failed : ", e)

    # ChatGPT 실패 시 ClovaX 호출
    try:
        clovax_response = await clovax_api(request)
        return clovax_response
    except Exception as e:
        print("ClovaX call failed : ", e)

    # ClovaX 실패 시 Gemini 호출
    try:
        gemini_response = await gemini_api(request)
        return gemini_response
    except Exception as e:
        print("Gemini call failed : ", e)
        # 모든 시도가 실패한 경우
        raise HTTPException(status_code=500, detail="All providers failed")

# 테스트 페이지 엔드포인트
@app.get("/test")
async def test(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})