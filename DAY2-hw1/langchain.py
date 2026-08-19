import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

# --- 1. 設定模型 (vLLM) ---
model = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    temperature=0,
    max_tokens=256
)

# --- 2. 定義人設分工 ---
# Branch A: 嚴肅 LinkedIn 專家
# Branch A: 嚴謹新聞主播
news_chain = (
    ChatPromptTemplate.from_template(
        "你是電視台的資深新聞主播。請針對主題：{topic}，用客觀、公式化且專業播報的口吻寫一段新聞快報(50字內)。"
    ) | model | StrOutputParser()
)

# Branch B: 網路迷因小編
meme_chain = (
    ChatPromptTemplate.from_template(
        "你是熟知網路梗的迷因小編。請針對主題：{topic}，用流行網路用語、梗圖口吻寫一段吐槽短評(50字內)。"
    ) | model | StrOutputParser()
)

combo_chain = RunnableParallel(
    news=news_chain,
    meme=meme_chain
)

# ==========================================
#  使用者輸入區 (只要改這裡，下面都會自動執行)
# ==========================================
target_topic = input("輸入主題:")#"Work Life Balance" 

#print(f"目標主題：{target_topic}\n")

# --- 模式 1: 體驗 Streaming (流式輸出) ---
# 適合場景：即時聊天，提升使用者體驗
#print(f"=== [Mode 1: Streaming] 即時生成中... ===")
#print("(觀察重點：不同欄位的文字會交錯出現)")

for chunk in combo_chain.stream({"topic": target_topic}):
    print(chunk) 

# --- 模式 2: 體驗 Batch (批次/完整輸出) ---
# 適合場景：後台排程，一次處理完畢再存檔
#print(f"\n\n=== [Mode 2: Batch] 完整執行結果 ===")
#print("(觀察重點：等待一段時間後，一次顯示完整結果)")

start_time = time.time()
# 雖然只有一個主題，但 Batch 介面要求輸入 List
results = combo_chain.batch([{"topic": target_topic}])
end_time = time.time()

# 取得第一個(也是唯一一個)結果
final_result = results[0]

print(f"耗時: {end_time - start_time:.2f} 秒")
print(f"--------------------------------------------------")
print(f"【LinkedIn 專家說】：\n{final_result['linkedin']}")
print(f"--------------------------------------------------")
print(f"【IG 網紅說】：\n{final_result['instagram']}")
print(f"--------------------------------------------------")