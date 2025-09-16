import os
import time
import logging
from typing import Any, Dict, List
from uuid import uuid4
import json

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)

PUBLIC_AGENT_CARD_PATH = '/.well-known/agent.json'
EXTENDED_AGENT_CARD_PATH = '/agent/authenticatedExtendedCard'

# 生成训练数据

async def get_agent_response(client: A2AClient, question: str) -> str:
    """Sends a question to the agent and returns the aggregated response."""
    parts = [{'kind': 'text', 'text': question}]
    multiturn_first: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': parts,
            'messageId': uuid4().hex,
            'metadata': {'language': "English", "user_id": 123456}
        },
    }
    streaming_request = SendStreamingMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(**multiturn_first)
    )
    stream_response = client.send_message_streaming(streaming_request)

    response_text = ""
    async for chunk in stream_response:
        # Assuming the relevant text is in a field named 'text' within a part of kind 'text'
        if chunk.delta and chunk.delta.parts:
            for part in chunk.delta.parts:
                if part.kind == 'text' and part.text:
                    response_text += part.text
    return response_text

async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    base_url = 'http://localhost:10066'

    async with httpx.AsyncClient(timeout=60.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        final_agent_card_to_use: AgentCard | None = None

        try:
            logger.info(f'Attempting to get Agent Card: {base_url}{PUBLIC_AGENT_CARD_PATH}')
            _public_card = await resolver.get_agent_card()
            final_agent_card_to_use = _public_card

            if _public_card.supportsAuthenticatedExtendedCard:
                try:
                    auth_headers_dict = {'Authorization': 'Bearer dummy-token-for-extended-card'}
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers_dict},
                    )
                    final_agent_card_to_use = _extended_card
                except Exception as e_extended:
                    logger.warning(f'Failed to get extended card: {e_extended}', exc_info=True)
        except Exception as e:
            logger.error(f'Failed to get AgentCard: {e}', exc_info=True)
            raise RuntimeError('Cannot get agent card, cannot continue.') from e

        client = A2AClient(httpx_client=httpx_client, agent_card=final_agent_card_to_use)
        logger.info('A2AClient initialized.')
        questions_file = './questions.txt'
        assert os.path.exists(questions_file), f'{questions_file} 问题文件不存在'
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]

        training_data: List[Dict[str, str]] = []
        for question in questions:
            logger.info(f"Processing question: {question}")
            answer = await get_agent_response(client, question)
            # The answer from the agent might be a JSON string, so we load it to format it nicely
            try:
                # Attempt to parse the answer as JSON, then dump it back to a string
                parsed_answer = json.loads(answer)
                formatted_answer = json.dumps(parsed_answer, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                # If it's not a valid JSON string, use the raw answer
                formatted_answer = answer
            
            training_data.append({"question": question, "answer": formatted_answer})
            logger.info(f"Received answer for question: {question}")


        with open('../example_one_data.json', 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info("Successfully generated training data and saved to example_one_data.json")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
