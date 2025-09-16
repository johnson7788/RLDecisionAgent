
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
    SendStreamingMessageRequest,
)

PUBLIC_AGENT_CARD_PATH = '/.well-known/agent.json'
EXTENDED_AGENT_CARD_PATH = '/agent/authenticatedExtendedCard'

async def get_agent_response(client: A2AClient, question: str) -> str:
    """Sends a question to the agent and returns the aggregated response based on the A2A protocol."""
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

    tool_outputs = []
    text_response = ""

    async for chunk in stream_response:
        chunk_data = chunk.model_dump(mode='json', exclude_none=True)
        
        result = chunk_data.get("result")
        if not result or result.get("kind") != "status-update":
            continue

        status = result.get("status")
        if not status:
            continue
        
        message = status.get("message")
        if not message:
            continue

        parts = message.get("parts")
        if not parts:
            continue

        for part in parts:
            if part.get("kind") == "data":
                data1 = part.get("data")
                if data1 and data1.get("data"):
                    for item in data1.get("data"):
                        if item and item.get("tool_output") is not None:
                            tool_outputs.append(item.get("tool_output"))
            elif part.get("kind") == "text":
                text = part.get("text")
                if text:
                    text_response += text

    final_result = None
    if tool_outputs:
        if len(tool_outputs) == 1:
            final_result = tool_outputs[0]
        else:
            final_result = tool_outputs
    elif text_response:
        final_result = text_response

    if final_result is not None:
        try:
            # If the result is a string that is valid JSON, parse it to embed it as an object.
            parsed_result = json.loads(final_result)
            return json.dumps({"result": parsed_result}, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            # Otherwise, treat as a plain string.
            return json.dumps({"result": final_result}, ensure_ascii=False)

    return json.dumps({})

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

        with open(QUESTION_FILE, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        questions = questions[:2]
        training_data: List[Dict[str, str]] = []
        for question in questions:
            logger.info(f"Processing question: {question}")
            answer = await get_agent_response(client, question)
            
            try:
                # Attempt to parse the answer as JSON, then dump it back to a string with indentation
                parsed_answer = json.loads(answer)
                formatted_answer = json.dumps(parsed_answer, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                # If it's not a valid JSON string, use the raw answer
                formatted_answer = answer
            
            training_data.append({"question": question, "answer": formatted_answer})
            logger.info(f"Received answer for question: {question}")


        with open(OUTLINE_DATA, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info("Successfully generated training data and saved to example_one_data.json")


if __name__ == '__main__':
    import asyncio
    QUESTION_FILE ="./questions.txt"
    OUTLINE_DATA="train.jsonl"
    asyncio.run(main())
