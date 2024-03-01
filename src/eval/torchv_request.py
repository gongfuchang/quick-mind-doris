import logging

import requests
import json

url = "https://demo.torchv.com/kl/api/saas/chat/completion"
headers = {
    "token": "b504e56b-023c-4647-acf8-bd8cc4d33203",
    "Content-Type": "application/json",
}

def extract_answer(response_json):
    choice = response_json.get("choices", [])
    if len(choice) == 0:
        return ""
    if choice[0]["message"]["role"] == "assistant" and choice[0]["message"]["content"] != "":
        return choice[0]["message"]["content"]
    return ""
def extract_contexts(response_json):
    paragraphs = response_json.get("paragraphs", [])
    if len(paragraphs) > 0:
        return [para["content"] for para in paragraphs]
    return []
def request_torchv(query):
    body = {
        "prompt": query,
        "conversationId": "",
        "stream": True,
        "status": True
    }
    data_buffer = ""
    with requests.post(url, stream=True, data=json.dumps(body), headers=headers) as r:
        answer = []
        contexts = []
        for line in r.iter_lines():
            # filter out keep-alive new lines and 'ping' messages
            if line:
                decoded_line = line.decode('utf-8')
                # print(decoded_line)

                if decoded_line.startswith('ping') or decoded_line.startswith('id:') or decoded_line.startswith('event:'):
                    continue
                try:
                    if decoded_line.startswith('data:'):
                        if data_buffer:
                            response_json = json.loads(data_buffer)
                            # print(response_json)
                            # Extract the necessary information from the response
                            answer.append(extract_answer(response_json))
                            contexts = extract_contexts(response_json)

                            data_buffer = ""
                        data_buffer += decoded_line.replace('data:', '').strip()
                    else:
                        data_buffer += decoded_line
                except Exception as e:
                    logging.info(f"Error in processing data: {e}")

        if data_buffer:
            response_json = json.loads(data_buffer)
            answer.append(extract_answer(response_json))
            contexts = extract_contexts(response_json)

        # Initialize the result JSON
        result_json = {
            "question": body["prompt"],
            "contexts": contexts,
            "answer": "".join(answer),
        }
        print(result_json)
        return r, result_json