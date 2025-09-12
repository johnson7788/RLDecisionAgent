#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/12 13:32
# @File  : llm_request.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
import os
import dotenv
import logging
from openai import OpenAI
dotenv.load_dotenv()
logging.basicConfig(level=logging.DEBUG)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="http://127.0.0.1:7300/v1")

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'user', 'content': "抛砖引玉是什么意思呀"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content)