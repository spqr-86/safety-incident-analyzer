import logging
from typing import Dict, TypedDict, Optional

from langchain_core.messages import HumanMessage
from src.llm_factory import get_gemini_llm, get_llm
from src.parsers import extract_text, parse_json_from_response
from src.prompt_manager import PromptManager
from src.types import RouteType
from config.settings import settings

logger = logging.getLogger(__name__)


class RouterOutput(TypedDict):
    type: RouteType
    response: Optional[str]  # For chitchat/out_of_scope


class RouterAgent:
    """
    Classifies user queries into:
    - rag_simple (direct factual question)
    - rag_complex (multi-step reasoning)
    - chitchat
    - out_of_scope
    """

    def __init__(self, llm_provider: str = "gemini"):
        self.prompt_manager = PromptManager()

        # Use fast model for routing (Flash)
        if llm_provider == "gemini":
            # No thinking needed for classification, just instruction following
            self.llm = get_gemini_llm(
                model_name=settings.GEMINI_FAST_MODEL,
                temperature=0.0,
                response_mime_type="application/json",
            )
        else:
            self.llm = get_llm()

    def route(self, query: str) -> RouterOutput:
        prompt = self.prompt_manager.render("router_v2", query=query)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            text = extract_text(response.content)
            parsed = parse_json_from_response(text)

            route_type_str = parsed.get("type", "rag_complex")

            # Map string to enum
            if route_type_str == "rag_simple":
                route_type = RouteType.RAG_SIMPLE
            elif route_type_str == "rag_complex":
                route_type = RouteType.RAG_COMPLEX
            elif route_type_str == "chitchat":
                route_type = RouteType.CHITCHAT
            elif route_type_str == "out_of_scope":
                route_type = RouteType.OUT_OF_SCOPE
            else:
                # Fallback
                logger.warning(
                    f"Unknown route type: {route_type_str}, defaulting to RAG_COMPLEX"
                )
                route_type = RouteType.RAG_COMPLEX

            return {"type": route_type, "response": parsed.get("response")}

        except Exception as e:
            logger.error(f"Router failed: {e}", exc_info=True)
            # Safe fallback
            return {"type": RouteType.RAG_COMPLEX, "response": None}
