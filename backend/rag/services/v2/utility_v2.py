import os
import re
import time
import json
import logging
from openai import OpenAI
from google import genai
from google.genai import types
from ollama import chat
from rag.services.utility import _get_available_key, _mark_exhausted, GEMINI_MODEL, call_openrouter, call_groq, OPENROUTER_MODEL, OPENROUTER_API_KEY
from rag.services.rag_tool import OLLAMA_RAG_SEARCH_TOOL_SCHEMA, RAG_SEARCH_TOOL_SCHEMA, execute_rag_search

logger = logging.getLogger(__name__)

OLLAMA_MODEL_TOOLS = os.environ.get("OLLAMA_MODEL_TOOLS", "qwen3.5:0.8b")


def clean_response(text: str) -> str:
    """Strip Qwen <think>...</think> blocks and stray XML tool-call artifacts."""
    if not text:
        return text
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?tool_call>', '', text)
    return text.strip()

# OpenRouter client (openai SDK pointed at OpenRouter)
_openrouter_client = None
if OPENROUTER_API_KEY:
    _openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

def call_gemini_with_tools(messages, crop_plan_context=None, max_retries=3):
    """
    Native Tool Calling flow for Gemini 2.0 Flash.
    """
    for attempt in range(max_retries):
        key = _get_available_key()
        if not key:
            print("All Gemini keys exhausted, waiting 5s...")
            time.sleep(5)
            continue

        try:
            client = genai.Client(api_key=key)

            config = types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.9,
                tools=[RAG_SEARCH_TOOL_SCHEMA],
                system_instruction=messages[0]["content"] if messages[0]["role"] == "system" else None,
            )

            # Convert standard messages to Gemini format
            gemini_messages = []
            for m in messages:
                if m["role"] != "system":
                    gemini_messages.append(types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part.from_text(text=m["content"])]))

            gemini_waterfall_str = os.environ.get("GEMINI_WATERFALL", "gemini-3.5-flash-lite,gemini-3.1-flash,gemini-3.1-pro")
            gemini_waterfall = [m.strip() for m in gemini_waterfall_str.split(",") if m.strip()]

            for gemini_model in gemini_waterfall:
                try:
                    print(f"\n🟢 [LLM ENGINE V2] Asking Gemini ({gemini_model}) to evaluate query...")
                    response = client.models.generate_content(
                        model=gemini_model,
                        contents=gemini_messages,
                        config=config
                    )

                    if response.function_calls:
                        function_call = response.function_calls[0]
                        if function_call.name == "rag_search":
                            query_arg = function_call.args.get("query", "")
                            print(f"🛠️  [TOOL CALL] rag_search(query='{query_arg}')")

                            chunks = execute_rag_search(query_arg)
                            
                            context_str = json.dumps(chunks)
                            gemini_messages.append(
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(text=f"Tool 'rag_search' returned this context:\n{context_str}\n\nPlease answer the original query using ONLY this context.")]
                                )
                            )

                            system_msg = next((m for m in messages if m.get("role") == "system"), None)
                            sys_text = system_msg["content"] if system_msg else ""
                            config_no_tools = types.GenerateContentConfig(
                                system_instruction=sys_text
                            )
                            print(f"🟢 [LLM ENGINE V2] Generating final response with retrieved chunks...")
                            final_response = client.models.generate_content(
                                model=gemini_model,
                                contents=gemini_messages,
                                config=config_no_tools
                            )
                            
                            try:
                                final_text = final_response.text or "I apologize, I could not generate a clear answer."
                            except Exception:
                                final_text = "I apologize, I could not generate a clear answer."
                            
                            return {
                                "answer": final_text,
                                "rewritten_query": query_arg,
                                "chunks": chunks
                            }

                    return {
                        "answer": response.text,
                        "rewritten_query": None,
                        "chunks": []
                    }

                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "503" in err_str or "quota" in err_str or "404" in err_str or "400" in err_str:
                        logger.warning(f"Gemini {gemini_model} unavailable or rate-limited, falling back to next model...")
                        print(f"⚠️  Gemini {gemini_model} unavailable or rate-limited, falling back to next model in bucket...")
                        continue
                    else:
                        logger.error(f"Gemini V2 Error ({gemini_model}): {err_str}")
                        raise e  # Unrelated error, try next key

            # If we get here, all models in the waterfall hit 429 for this specific key
            _mark_exhausted(key, cooldown=30)
            time.sleep(1)

        except Exception as e:
            # Handle non-quota errors that bubbled up
            pass

    return None

def call_openrouter_with_tools(messages, crop_plan_context=None, max_retries=3):
    """
    OpenRouter Tool Calling via openai SDK (official pattern).
    Two-phase approach: first call with tool_choice="auto",
    second call with tool_choice="none" to force a final answer.
    """
    if not _openrouter_client:
        logger.warning("No OPENROUTER_API_KEY configured — skipping OpenRouter V2")
        return None

    # Build the tool spec (same schema used across all providers)
    tools = [OLLAMA_RAG_SEARCH_TOOL_SCHEMA]

    # Tool mapping: name -> local function
    def local_rag_search(query=""):
        return execute_rag_search(query)

    TOOL_MAPPING = {
        "rag_search": local_rag_search,
    }

    for attempt in range(max_retries):
        try:
            # Convert internal messages to OpenAI-compatible dicts
            openrouter_messages = []
            for m in messages:
                openrouter_messages.append({"role": m["role"], "content": m["content"]})

            print(f"\n🟣 [LLM ENGINE V2] Asking OpenRouter ({OPENROUTER_MODEL}) to evaluate query...")

            # ── Phase 1: Let model decide whether to call tools ──
            final_content = ""
            chunks = []
            rewritten_query = None
            tool_was_called = False

            resp = _openrouter_client.chat.completions.create(
                model=OPENROUTER_MODEL,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=4096,
                messages=openrouter_messages,
                parallel_tool_calls=False,
            )

            choice = resp.choices[0]

            # Append assistant message to conversation
            msg_to_append = {"role": "assistant", "content": choice.message.content}
            if choice.message.tool_calls:
                msg_to_append["content"] = None
                msg_to_append["tool_calls"] = [tc.model_dump() for tc in choice.message.tool_calls]
            openrouter_messages.append(msg_to_append)

            tool_calls = choice.message.tool_calls
            if not tool_calls:
                # Model answered directly — no tool needed
                final_content = clean_response(choice.message.content or "")
            else:
                # Execute tools (with deduplication)
                seen_tool_calls = set()
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args_str = tool_call.function.arguments
                    dedup_key = (tool_name, tool_args_str)
                    if dedup_key in seen_tool_calls:
                        print(f"⚠️  Skipping duplicate tool call: {tool_name}")
                        continue
                    seen_tool_calls.add(dedup_key)

                    tool_args = json.loads(tool_args_str)
                    print(f"🛠️  [TOOL CALL] {tool_name}({tool_args})")

                    tool_result = TOOL_MAPPING[tool_name](**tool_args)

                    if tool_name == "rag_search" and not rewritten_query:
                        rewritten_query = tool_args.get("query", "")
                        chunks = tool_result

                    openrouter_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_result),
                    })
                tool_was_called = True

            # ── Phase 2: Force final answer (no more tool calls) ──
            if tool_was_called:
                resp2 = _openrouter_client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    tools=tools,
                    tool_choice="none",
                    temperature=0.2,
                    max_tokens=4096,
                    messages=openrouter_messages,
                )
                final_content = clean_response(resp2.choices[0].message.content or "")

            if not final_content.strip():
                raise ValueError("OpenRouter returned empty final content")

            print(f"🟣 [LLM ENGINE V2] OpenRouter response received ({len(final_content)} chars)")
            return {
                "answer": final_content.strip(),
                "rewritten_query": rewritten_query,
                "chunks": chunks,
            }

        except Exception as e:
            logger.error(f"OpenRouter V2 Error: {e}")
            print(f"OpenRouter V2 Error: {e}")
            time.sleep(1)

    return None

# Groq client (openai SDK pointed at Groq)
_groq_client = None
if os.environ.get("GROQ_API_KEY"):
    _groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )

def call_groq_with_tools(messages, crop_plan_context=None, max_retries=3):
    """
    Groq Tool Calling via openai SDK.
    Two-phase approach: first call with tool_choice="auto",
    second call with tool_choice="none" to force a final answer.
    """
    if not _groq_client:
        logger.warning("No GROQ_API_KEY configured — skipping Groq V2")
        return None

    tools = [OLLAMA_RAG_SEARCH_TOOL_SCHEMA]
    groq_waterfall_str = os.environ.get("GROQ_WATERFALL", "qwen/qwen3.8-27b,openai/gpt-oss-20b,openai/gpt-oss-120b")
    groq_waterfall = [m.strip() for m in groq_waterfall_str.split(",") if m.strip()]

    def local_rag_search(query=""):
        return execute_rag_search(query, top_k=5)

    TOOL_MAPPING = {
        "rag_search": local_rag_search,
    }

    for groq_model in groq_waterfall:
        for attempt in range(max_retries):
            try:
                # ── Groq-Specific History Filtering ──
                # Only keep system prompt, up to 2 previous user queries (n=2, no answers), and current query
                system_msg = next((m for m in messages if m["role"] == "system"), None)
                current_query = messages[-1]
            
                # Extract history excluding system and current
                history = [m for m in messages if m != system_msg and m != current_query]
            
                # Keep the last 2 messages (usually 1 User and 1 Assistant) to maintain conversational context
                # without blowing up the Groq input token limit.
                recent_history = history[-2:]
            
                groq_messages = []
                if system_msg:
                    groq_messages.append({"role": system_msg["role"], "content": system_msg["content"]})
            
                for m in recent_history:
                    groq_messages.append({"role": m["role"], "content": m["content"]})
                
                groq_messages.append({"role": current_query["role"], "content": current_query["content"]})

                print(f"\n🟡 [LLM ENGINE V2] Asking Groq ({groq_model}) to evaluate query...")

                # ── Phase 1: Let model decide whether to call tools ──
                final_content = ""
                chunks = []
                rewritten_query = None
                tool_was_called = False

                resp = _groq_client.chat.completions.create(
                    model=groq_model,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=200,
                    messages=groq_messages,
                    parallel_tool_calls=False,
                )

                choice = resp.choices[0]

                # Append assistant message to conversation
                msg_to_append = {"role": "assistant", "content": choice.message.content}
                if choice.message.tool_calls:
                    msg_to_append["content"] = None
                    msg_to_append["tool_calls"] = [tc.model_dump() for tc in choice.message.tool_calls]
                groq_messages.append(msg_to_append)

                tool_calls = choice.message.tool_calls
                if not tool_calls:
                    # Model answered directly — no tool needed
                    final_content = clean_response(choice.message.content or "")
                else:
                    # Execute tools (with deduplication)
                    seen_tool_calls = set()
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments
                        dedup_key = (tool_name, tool_args_str)
                        if dedup_key in seen_tool_calls:
                            print(f"⚠️  Skipping duplicate tool call: {tool_name}")
                            continue
                        seen_tool_calls.add(dedup_key)

                        tool_args = json.loads(tool_args_str)
                        print(f"🛠️  [TOOL CALL] {tool_name}({tool_args})")

                        tool_result = TOOL_MAPPING[tool_name](**tool_args)

                        if tool_name == "rag_search" and not rewritten_query:
                            rewritten_query = tool_args.get("query", "")
                            chunks = tool_result

                        groq_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": json.dumps(tool_result),
                        })
                    tool_was_called = True

                # ── Phase 2: Force final answer (no more tool calls) ──
                if tool_was_called:
                    resp2 = _groq_client.chat.completions.create(
                        model=groq_model,
                        tools=tools,
                        tool_choice="none",
                        temperature=0.2,
                        max_tokens=800,
                        messages=groq_messages,
                    )
                    final_content = clean_response(resp2.choices[0].message.content or "")

                if not final_content.strip():
                    raise ValueError("Groq returned empty final content")

                print(f"🟡 [LLM ENGINE V2] Groq response received ({len(final_content)} chars)")
                return {
                    "answer": final_content.strip(),
                    "rewritten_query": rewritten_query,
                    "chunks": chunks,
                }

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str or "400" in err_str or "404" in err_str:
                    logger.warning(f"Groq {groq_model} unavailable or rate-limited, falling back to next model...")
                    print(f"⚠️  Groq {groq_model} rate-limited or failed, falling back to next model...")
                    break # Break out of attempt loop, move to next model
            
                logger.error(f"Groq V2 Error ({groq_model}): {e}")
                print(f"Groq V2 Error ({groq_model}): {e}")
                time.sleep(1)

    return None

def call_ollama_with_tools(messages, crop_plan_context=None, max_retries=3):
    """
    Ollama Tool Calling flow for local models (e.g., qwen3.5:0.8b) using python client.
    """
    for attempt in range(max_retries):
        try:
            ollama_messages = []
            for m in messages:
                ollama_messages.append({"role": m["role"], "content": m["content"]})
                
            # Force tiny models to use the tool by injecting instruction directly into the latest message
            original_last_query = ""
            if ollama_messages and ollama_messages[-1]["role"] == "user":
                original_last_query = ollama_messages[-1]["content"]
                ollama_messages[-1]["content"] += "\n\n[SYSTEM INSTRUCTION: MUST use rag_search tool to answer this!]"
            
            print(f"\n🟢 [LLM ENGINE V2] Asking Ollama ({OLLAMA_MODEL_TOOLS}) to evaluate query...")
            
            response = chat(
                model=OLLAMA_MODEL_TOOLS,
                messages=ollama_messages,
                tools=[OLLAMA_RAG_SEARCH_TOOL_SCHEMA],
                think=False,
                stream=False,
                options={
                    "temperature": 0.2,
                    "top_p": 0.9
                }
            )
            
            if response.message.tool_calls:
                tool_call = response.message.tool_calls[0]
                if tool_call.function.name == "rag_search":
                    query_arg = tool_call.function.arguments.get("query", "")
                    print(f"🛠️  [TOOL CALL] rag_search(query='{query_arg}')")

                    chunks = execute_rag_search(query_arg)
                    
                    assistant_msg = {
                        "role": response.message.role,
                        "content": response.message.content or ""
                    }
                    if response.message.tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in response.message.tool_calls
                        ]
                    
                    if original_last_query and ollama_messages[-1]["role"] == "user":
                        ollama_messages[-1]["content"] = original_last_query
                        
                    ollama_messages.append(assistant_msg)
                    
                    formatted_context = ""
                    for i, c in enumerate(chunks):
                        doc = c.get("document_name", f"Doc_{i}")
                        text = c.get("text", "")
                        formatted_context += f"[Source: {doc}]\n{text}\n\n"
                    
                    ollama_messages.append({
                        "role": "tool",
                        "content": formatted_context if formatted_context else "No results found.",
                        "name": "rag_search"
                    })
                    
                    print(f"🟢 [LLM ENGINE V2] Generating final response with retrieved chunks...")
                    
                    final_response = chat(
                        model=OLLAMA_MODEL_TOOLS,
                        messages=ollama_messages,
                        think=False,
                        stream=False,
                        options={
                            "temperature": 0.2,
                            "top_p": 0.9,
                            "num_ctx": 16384
                        }
                    )
                    
                    return {
                        "answer": clean_response(final_response.message.content).strip(),
                        "rewritten_query": query_arg,
                        "chunks": chunks
                    }
                    
            print("⚠️  [LLM ENGINE V2] Model chose NOT to use the tool. It is generating the response from memory alone.")
            return {
                "answer": clean_response(response.message.content).strip(),
                "rewritten_query": None,
                "chunks": []
            }
            
        except Exception as e:
            logger.error(f"Ollama V2 Error: {e}")
            time.sleep(1)
            
    return None

# =======================================================
# Dynamic Priority Queue for LLM Routing (JSON Persistent)
# =======================================================
DEFAULT_PRIORITY = ["GEMINI", "OPENROUTER", "GROQ", "OLLAMA"]
RESET_INTERVAL = 24 * 3600  # 24 hours in seconds
QUEUE_STATE_FILE = os.path.join(os.path.dirname(__file__), "llm_queue_state.json")

def _load_queue_state():
    if os.path.exists(QUEUE_STATE_FILE):
        try:
            with open(QUEUE_STATE_FILE, "r") as f:
                data = json.load(f)
                return data.get("priority", list(DEFAULT_PRIORITY)), data.get("last_reset_time", time.time())
        except Exception as e:
            logger.error(f"Error reading queue state: {e}")
    
    # First time or error: return defaults and save
    default_time = time.time()
    _save_queue_state(list(DEFAULT_PRIORITY), default_time)
    return list(DEFAULT_PRIORITY), default_time

def _save_queue_state(priority, last_reset_time):
    try:
        with open(QUEUE_STATE_FILE, "w") as f:
            json.dump({
                "priority": priority,
                "last_reset_time": last_reset_time
            }, f)
    except Exception as e:
        logger.error(f"Error saving queue state: {e}")

def get_engine_queue():
    priority, last_reset_time = _load_queue_state()
    
    # Reset order every 24 hours
    if time.time() - last_reset_time > RESET_INTERVAL:
        priority = list(DEFAULT_PRIORITY)
        last_reset_time = time.time()
        _save_queue_state(priority, last_reset_time)
        logger.info("LLM Priority Queue has been reset (24h refresh).")
        
    # Filter out engines disabled in .env
    active_queue = []
    for engine in priority:
        env_flag = f"ENABLE_{engine}"
        if os.environ.get(env_flag, "true").lower() == "true":
            active_queue.append(engine)
            
    return active_queue

def mark_engine_exhausted(engine_name):
    priority, last_reset_time = _load_queue_state()
    if engine_name in priority:
        priority.remove(engine_name)
        priority.append(engine_name) # Move to back of the line
        _save_queue_state(priority, last_reset_time)
        logger.warning(f"Engine {engine_name} exhausted/failed. Moved to back of priority queue.")


def call_llm_v2(messages, crop_plan_context=None):
    """
    Unified Waterfall Router for V2 using a Dynamic Priority Queue.
    If an engine fails, it moves to the back of the line.
    """
    print("\n" + "="*50)
    
    queue = get_engine_queue()
    print(f"🚦 Current LLM Priority Queue: {' -> '.join(queue)}")
    
    for engine in queue:
        res = None
        if engine == "GEMINI":
            res = call_gemini_with_tools(messages, crop_plan_context)
        elif engine == "OPENROUTER":
            res = call_openrouter_with_tools(messages, crop_plan_context)
        elif engine == "GROQ":
            res = call_groq_with_tools(messages, crop_plan_context)
        elif engine == "OLLAMA":
            res = call_ollama_with_tools(messages, crop_plan_context)
            
        if res:
            return res
        else:
            # Engine failed or returned None, drop its priority
            mark_engine_exhausted(engine)
            print(f"⚠️  {engine} failed. Trying next engine in queue...")
            
    logger.error("All LLM engines in the active queue failed in V2.")
    return None
