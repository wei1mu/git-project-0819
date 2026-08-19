import os
import re
import json
import base64
import operator
from typing import Annotated, List, TypedDict, Union, Literal
from datetime import datetime

# LangChain / LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import requests
from playwright.sync_api import sync_playwright

# --- 設定配置 ---
llm = ChatOpenAI(
    base_url="https://3090p8000.huannago.com/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="google/gemma-3-27b-it",
    temperature=0.5,
)

# 快取檔案名稱
CACHE_FILE = "qa_cache.json"

SAVE_VLM_OUTPUT = True  # 開關：設為 True 則儲存圖片與文字，False 則不存
VLM_LOG_DIR = "vlm_logs" # 儲存的根目錄

# --- 0. 快取工具函數 ---

def get_clean_key(text: str) -> str:
    """統一將問題標準化：去除空白與問號，作為 Cache 的 Key"""
    return text.replace(" ", "").replace("?", "")

def load_cache():
    """從 JSON 讀取快取資料"""
    if not os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
        return {}
    
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_cache(new_data: dict):
    """將資料寫入 JSON"""
    current_data = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        except:
            pass
            
    current_data.update(new_data)
    
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)

# --- 1. 定義狀態 (State) ---
class AgentState(TypedDict):
    query: str                                      # 使用者原始問題
    knowledge_base: Annotated[List[str], operator.add] # 收集到的資訊
    search_history: Annotated[List[str], operator.add] # 搜尋過的關鍵字
    round_count: int                                # 當前輪數
    status: str                                     # "CONTINUE" or "DONE"
    latest_thought: str                             # 為了讓下一步知道當前的思考
    final_output: str                               # 最終答案
    source: str                                     # 來源標記: "CACHE" 或 "GENERATED"
    resolved_facts: str                             # 經過邏輯推理後的一致性事實

# --- 2. 定義結構化輸出 (Pydantic Models) ---
class PlanDecision(BaseModel):
    thought: str = Field(description="分析資訊缺口，說明下一步策略")
    status: Literal["CONTINUE", "DONE"] = Field(description="決定是否繼續搜尋")

class SearchQuery(BaseModel):
    keywords: str = Field(description="搜尋引擎用的關鍵字")
    intent: str = Field(description="查詢意圖")
    time_range: Literal["day", "week", "month", "year", "all"] = Field(default="all")

class ReasoningOutput(BaseModel):
    timeline: str = Field(description="根據資料重建的事件時間軸 (含推論年份)")
    conflict_analysis: str = Field(description="分析資料中的矛盾點")
    resolved_content: str = Field(description="經過去除矛盾、補全邏輯後的最終事實摘要")

class RelevanceCheck(BaseModel):
    status: Literal["RELEVANT", "IRRELEVANT"]

# --- 4. 工具函數 (Tools) ---

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")[:50]

def vlm_read_website(url: str, title: str, query_context: str = "general") -> str:
    print(f"    📸 [VLM] 啟動視覺閱讀: {title}")
    
    safe_title = sanitize_filename(title)
    safe_query = get_clean_key(query_context)[:30]
    timestamp = datetime.now().strftime("%H%M%S")
    
    save_dir = os.path.join(VLM_LOG_DIR, safe_query, f"{timestamp}_{safe_title}")
    
    if SAVE_VLM_OUTPUT and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def capture_rolling_screenshots(url):
        screenshots_b64 = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
                context = browser.new_context(viewport={'width': 1280, 'height': 1200})
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=10000)
                page.wait_for_timeout(3000)
                
                page.add_style_tag(content="""
                    iframe { opacity: 0 !important; pointer-events: none !important; }
                    div[id*='cookie'], div[class*='cookie'], div[id*='ads'], div[class*='ads'] { display: none !important; }
                    div[class*='overlay'], div[id*='overlay'], div[class*='popup'] { opacity: 0 !important; pointer-events: none !important; }
                """)

                total_height = page.evaluate("document.body.scrollHeight")
                viewport_height = 1200
                current_scroll = 0
                
                scroll_count = 0
                for _ in range(5):
                    page.evaluate(f"window.scrollTo(0, {current_scroll})")
                    page.wait_for_timeout(1000)
                    
                    screenshot_bytes = page.screenshot()
                    b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                    screenshots_b64.append(b64)
                    
                    if SAVE_VLM_OUTPUT:
                        img_path = os.path.join(save_dir, f"scan_{scroll_count}.png")
                        with open(img_path, "wb") as f:
                            f.write(screenshot_bytes)

                    current_scroll += (viewport_height - 200)
                    scroll_count += 1
                    if current_scroll >= total_height: break
                    
                browser.close()
        except Exception as e:
            print(f"    ❌ 截圖失敗: {e}")
        return screenshots_b64

    images = capture_rolling_screenshots(url)
    if not images: return "無法讀取網頁內容"
    
    keywords = query_context if 'query_context' in locals() else "未知"

    prompt_text = f"""你是一個高精度的資訊提取員。
網頁標題：{title}
搜尋關鍵字：{keywords}

任務目標：請依據「搜尋關鍵字」從圖片中篩選並提取最相關的資訊。

執行準則：
1. **聚焦關鍵字**：只提取與關鍵字直接相關的內容，忽略無關的旁支末節。
2. **排除雜訊**：嚴格忽略廣告、導覽列、彈窗與側邊欄推薦。
3. **數據優先**：若圖片中包含「圖表」或「表格」，請務必轉述其中的具體數值、趨勢與時間點。"""

    msg_content = [{"type": "text", "text": prompt_text}]
    for img in images:
        msg_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

    response = ""

    try:
        response_msg = llm.invoke([HumanMessage(content=msg_content)])
        response = response_msg.content
        
        if SAVE_VLM_OUTPUT:
            txt_path = os.path.join(save_dir, "vlm_summary.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"URL: {url}\n")
                f.write(f"Title: {title}\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write("-" * 20 + "\n")
                f.write(response)
            print(f"      💾 VLM 資料已備份至: {save_dir}")
            
    except Exception as e:
        print(e)
    return response

def search_searxng(query: str, time_range: str = None):
    params = {"q": query, "format": "json", "language": "zh-TW"}
    if time_range and time_range != "all": params["time_range"] = time_range
    try:
        SEARXNG_URL = "https://puli-8080.huannago.com/search"
        res = requests.get(SEARXNG_URL, params=params).json()
        return res.get('results', [])[:3]
    except Exception as e:
        print(f"Search Error: {e}")
        return []

# --- 5. 定義節點 (Nodes) ---

def check_cache_node(state: AgentState):
    """檢查快取節點"""
    print(f"\n🔎 [Cache] 檢查快取: {state['query']}")
    cache_data = load_cache()
    key = get_clean_key(state['query'])
    
    if key in cache_data:
        print("    🎉 命中快取！直接返回結果。")
        return {
            "final_output": cache_data[key],
            "status": "DONE",
            "source": "CACHE"
        }
    else:
        print("    💨 未命中快取，進入 Agent 思考流程。")
        return {"source": "GENERATED"}

def planner_node(state: AgentState):
    """思考節點：決定下一步"""
    print(f"\n🧠 [Think] Round {state['round_count']}")

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    
    kb_text = "\n".join(state['knowledge_base'])
    history_text = ", ".join(state['search_history'])
    
    system_prompt = f"""你是一名專業的調查研究員，擅長快速掌握陌生領域的知識。
你將會接收到以下資訊:
使用者原始問題: (使用者q)
目前已收集的資訊: (資訊)
    
過去已搜尋過的關鍵字: (關鍵字集合)

【核心思維原則】
1. 多方求證：尋找第二來源佐證。
2. 結構化思考：從多個維度進行搜集。
3. 時效性：注意資訊的發布時間，必須判斷是否落後於當前時間。
4. 懷疑論：必須尋找客觀數據來驗證內容。
5. 靈活應變：必須以不同面向的思路去判斷資訊是否足以回應使用者原始問題。
6. 狀態：目前已收集的資訊足以回應使用者原始問題，則可DONE，反之則CONTINUE。若明確缺乏某相關資訊則絕對禁止DONE。

【當前基準資訊】
現在時間: {current_date}
現在年份: {current_year}

## 使用者問題: {state['query']}

## 已收集資訊:
---
{kb_text if kb_text else "(無)"}

## 知識脈絡:
{state['resolved_facts']}

## 已搜關鍵字: {history_text}

請決定是否資訊充足(DONE)或需要繼續搜尋(CONTINUE)。"""

    structured_llm = llm.with_structured_output(PlanDecision)
    decision = structured_llm.invoke([SystemMessage(content=system_prompt)])
    
    print(f"    💭 想法: {decision.thought}")
    return {
        "status": decision.status, 
        "latest_thought": decision.thought,
        "round_count": state['round_count'] + 1
    }

def query_gen_node(state: AgentState):
    """生成關鍵字節點"""
    history_str = ", ".join(state['search_history']) if state['search_history'] else "無"

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year

    system_prompt = f"""你是一個專業的搜尋查詢優化助手,負責將使用者的問題和當前思考脈絡轉換為最有效的搜尋引擎關鍵字。

## 輸入格式
你將收到兩項輸入:
1. 使用者問題 (q)
2. 當前目標思考內容

## 核心任務
根據以上兩項輸入,生成一組精確的搜尋引擎關鍵字。

## 關鍵字生成原則
- 簡潔性: 嚴格限制關鍵字總數在 1-3 個單詞以內。
- 去雜訊: 去除「原因」、「影響」等抽象名詞，僅保留實體與事件。
- 聚焦性: 生成最核心的關鍵字。
- 禁止重複: 絕對不能與歷史紀錄重複。

【當前基準資訊】
現在時間: {current_date}
現在年份: {current_year}

原問題: {state['query']}
當前策略: {state['latest_thought']}
已搜尋過的關鍵字: {history_str}"""

    structured_llm = llm.with_structured_output(SearchQuery)
    query_obj = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="")
    ])
    
    print(f"    🔑 關鍵字: {query_obj.keywords} (範圍: {query_obj.time_range})")
    return {"search_history": [query_obj.keywords]}

def search_process_node(state: AgentState):
    """搜尋 + VLM 閱讀節點"""
    current_keyword = state['search_history'][-1]
    results = search_searxng(current_keyword)
    new_knowledge = []
    
    relevance_llm = llm.with_structured_output(RelevanceCheck)
    
    for item in results:
        print(f"    🌐 訪問: {item['title']}")
        content = vlm_read_website(item['url'], item['title'], query_context=current_keyword)
        
        check_prompt = f"原問題: {state['query']}\n搜尋關鍵字: {current_keyword}\n網頁內容摘要: {content}\n這份內容是否對回答問題有幫助？"
        check = relevance_llm.invoke([HumanMessage(content=check_prompt)])
        
        if check.status == "RELEVANT":
            print("      ✅ 相關，已收錄")
            entry = f"來源: {item['title']}\n網址: {item['url']}\n內容: {content}\n---"
            new_knowledge.append(entry)
        else:
            print("      🗑️ 無關，跳過")
            
    return {"knowledge_base": new_knowledge}

def reasoning_node(state: AgentState):
    """邏輯推理與事實核查"""
    print(f"\n🧠 [Logic] 正在進行事實邏輯推理與時序重組...")
    
    kb_text = "\n".join(state['knowledge_base'])
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    
    system_prompt = f"""你是一名高階情報分析師，專門負責處理碎片化資訊的邏輯重組。
你的任務不是回答問題，而是「整理事實」給下游的寫作模型使用。

【當前基準資訊】
現在時間: {current_date}
現在年份: {current_year}

原始資料片段:
{kb_text}"""

    user_prompt = "請分析上述資料，重點解決時間與順序的邏輯問題，並輸出整理後的事實。"

    try:
        structured_llm = llm.with_structured_output(ReasoningOutput)
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        print(f"    📅 重建時間軸: {result.timeline}")
        print(f"    ⚔️ 邏輯修正: {result.conflict_analysis}")
        print(f"    ✅ 事實定案: {result.resolved_content[:50]}...")
        
        return {"resolved_facts": result.resolved_content}
        
    except Exception as e:
        print(f"    ❌ 推理節點發生錯誤: {e}，將使用原始資料。")
        return {"resolved_facts": ""}

def final_answer_node(state: AgentState):
    """總結節點 (含快取寫入)"""
    # 若已經是快取命中的狀態，直接返回已有結果
    if state.get("source") == "CACHE":
        return {"final_output": state["final_output"]}

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year

    if state['round_count'] > 3:
        print("已達到最大輪次，強制跳出")
    
    print("\n📝 [Summary] 正在撰寫最終報告...")

    context_data = state.get('resolved_facts', "")
    kb_text = "\n".join(state['knowledge_base'])
    
    prompt = f"""請根據以下收集到的資料，完整回答使用者的問題。
所回答的資訊來源需使用["id"]標記，並在結尾統一將該編號後補充其URL與標題。

【當前基準資訊】
現在時間: {current_date}
現在年份: {current_year}

## 問題: {state['query']}

## 資料庫:
---
{kb_text}

## 驗證後的事實資料:
{context_data}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    final_output = response.content

    # 將新產生的結果寫入快取
    print(f"    💾 將結果寫入快取檔案 ({CACHE_FILE})")
    save_cache({get_clean_key(state['query']): final_output})

    return {"final_output": final_output}

# --- 6. 建構圖 (Graph Construction) ---

workflow = StateGraph(AgentState)

# 1. 註冊所有節點
workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_process_node)
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("final_answer", final_answer_node)

# 2. 設定入口點為 check_cache
workflow.set_entry_point("check_cache")

# 3. 快取路由：命中直奔 final_answer，未命中進入 planner
def cache_router(state: AgentState):
    if state.get("source") == "CACHE":
        return "final_answer"
    return "planner"

workflow.add_conditional_edges(
    "check_cache",
    cache_router,
    {
        "final_answer": "final_answer",
        "planner": "planner"
    }
)

# 4. Planner 路由：DONE 或超過輪次進入 final_answer，否則繼續 query_gen
def planner_router(state: AgentState):
    if state['status'] == "DONE" or state['round_count'] > 3:
        return "gen_answer"
    return "continue"

workflow.add_conditional_edges(
    "planner",
    planner_router,
    {
        "continue": "query_gen",
        "gen_answer": "final_answer"
    }
)

# 5. 一般邊界連接
workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "reasoning")
workflow.add_edge("reasoning", "planner")
workflow.add_edge("final_answer", END)

# 編譯圖
app = workflow.compile()
print(app.get_graph().draw_ascii())

# --- 7. 執行 ---
if __name__ == "__main__":
    print(f"快取檔案: {os.path.abspath(CACHE_FILE)}")
    
    while True:
        user_input = input("\n請輸入問題 (輸入 q 離開): ")
        if user_input.lower() == 'q':
            break

        initial_state = {
            "query": user_input,
            "knowledge_base": [],
            "search_history": [],
            "round_count": 0,
            "status": "CONTINUE",
            "latest_thought": "",
            "final_output": "",
            "source": "GENERATED",
            "resolved_facts": ""
        }
        
        print(f"🚀 開始處理: {user_input}")
        
        final_state = app.invoke(initial_state)
        
        print("\n" + "="*30)
        print(f"[{final_state.get('source', 'UNKNOWN')}] 最終報告:")
        print(final_state.get("final_output", "未生成最終報告"))
        print("="*30)
