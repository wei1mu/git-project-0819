import os
import time
import requests
from pathlib import Path
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# ================= 配置區 =================
# 請填入 API Key
CONFIG = {
    "asr_api_url": "https://3090api.huannago.com",
    "asr_auth": ("nutc2504", "nutc2504") # 老師提供的帳密
}

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="EMPTY",
    model="google/gemma-3-27b-it",
    temperature=0
)

# ================= 0. ASR API 工具函式 (封裝 test.py) =================
# 這是將原本的 script 變成一個可呼叫的 function
def call_asr_api(audio_path: str):
    """上傳音檔並等待轉錄結果 (TXT + SRT)"""
    print(f"🎤 [ASR] 正在上傳音檔: {audio_path} ...")
    
    create_url = f"{CONFIG['asr_api_url']}/api/v1/subtitle/tasks"
    
    # 1. 上傳
    with open(audio_path, "rb") as f:
        r = requests.post(create_url, files={"audio": f}, timeout=60, auth=CONFIG['asr_auth'])
    r.raise_for_status()
    task_id = r.json()["id"]
    print(f"⏳ [ASR] 任務 ID: {task_id}，等待轉錄中...")

    # 2. 輪詢 (Polling) 直到完成
    txt_url = f"{CONFIG['asr_api_url']}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT"
    srt_url = f"{CONFIG['asr_api_url']}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"

    def wait_download(url):
        for _ in range(300): # 最多等 600秒
            try:
                resp = requests.get(url, timeout=10, auth=CONFIG['asr_auth'])
                if resp.status_code == 200:
                    return resp.text
            except:
                pass
            time.sleep(2)
        return None

    # 這裡我們依序下載 TXT 和 SRT
    txt_content = wait_download(txt_url)
    srt_content = wait_download(srt_url)
    
    if not txt_content or not srt_content:
        raise Exception("ASR 轉錄逾時或失敗")
        
    print("✅ [ASR] 轉錄完成！")
    return txt_content, srt_content

# ================= 1. 定義狀態 (State) =================
class MeetingState(TypedDict):
    audio_path: str         # 輸入：音檔路徑
    transcript_txt: str     # 中間產物：純文字稿
    transcript_srt: str     # 中間產物：時間軸稿
    minutes: str            # 輸出 A：詳細記錄
    summary: str            # 輸出 B：重點摘要
    final_report: str       # 最終產出：整合報告

# ================= 2. 定義節點 (Nodes) =================

def asr_node(state: MeetingState):
    """負責呼叫 ASR API 的節點"""
    txt, srt = call_asr_api(state["audio_path"])
    # 更新 State
    return {"transcript_txt": txt, "transcript_srt": srt}

def minutes_node(state: MeetingState):
    """(平行節點 A) 紀錄員：負責讀取 SRT 並整理成條列式記錄"""
    print("📝 [Minutes] 正在整理詳細會議記錄 (含時間軸)...")
    
    prompt = f"""
    你是專業的會議記錄員。請根據以下的 SRT 時間軸內容，整理出詳細的會議發言紀錄。
    格式要求：
    - 保留重要發言的時間點
    - 將口語轉化為書面語
    - 條列式呈現
    
    SRT 內容：
    {state['transcript_srt'][:2000]} ... (略)
    """
    # 注意：實際使用時若 SRT 太長可能要切分，這邊示範直接丟
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"minutes": response.content}

def summary_node(state: MeetingState):
    """(平行節點 B) 總結者：負責讀取 TXT 並撰寫高階摘要"""
    print("💡 [Summary] 正在撰寫高階摘要...")
    
    prompt = f"""
    你是公司的高階特助。請根據以下的會議逐字稿，寫一份 200 字以內的精簡摘要。
    專注於：決策結果、待辦事項(Action Items)。
    
    逐字稿內容：
    {state['transcript_txt'][:2000]} ... (略)
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content}

def writer_node(state: MeetingState):
    """(匯聚節點) 整合報告"""
    print("🖨️ [Writer] 正在整合最終報告...")
    
    report = f"""
# 📄 智慧會議紀錄報告

## 🎯 重點摘要 (Executive Summary)
{state['summary']}

---

## ⏱️ 詳細記錄 (Detailed Minutes)
{state['minutes']}

---
*本報告由 AI Agent 自動生成*
    """
    return {"final_report": report}

# ================= 3. 建構 Graph =================

workflow = StateGraph(MeetingState)

# 加入節點
workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_node)
workflow.add_node("summarizer", summary_node)
workflow.add_node("writer", writer_node)

# 設定流程
workflow.set_entry_point("asr")

# --- 關鍵：平行處理 (Fan-out) ---
# 從 asr 結束後，同時指向 minutes_taker 和 summarizer
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")

# --- 關鍵：匯聚 (Fan-in) ---
# 兩個平行節點都指向 writer
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")

workflow.add_edge("writer", END)

app = workflow.compile()
print(app.get_graph().draw_ascii())
# ================= 4. 執行測試 =================
if __name__ == "__main__":
    # 請確保資料夾內有測試音檔
    test_audio = "D:/Podcast_EP14_20s.wav"
    
    if not os.path.exists(test_audio):
        print(f"❌ 找不到音檔: {test_audio}，請確認路徑。")
    else:
        print("🚀 會議助手啟動中...")
        result = app.invoke({"audio_path": test_audio})
        
        print("\n" + "="*30)
        print(result["final_report"])
        
        # 也可以存檔
        with open("meeting_report.md", "w", encoding="utf-8") as f:
            f.write(result["final_report"])
            print(f"✅ 報告已儲存至 meeting_report.md")