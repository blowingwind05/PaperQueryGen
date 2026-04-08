from flask import Flask, jsonify, render_template_string, request
import pandas as pd
import random
import os
import csv
import datetime
import re
import json
import numpy as np
import requests
import queue
import threading

class ArxivJSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 NumPy 数组、Pandas 对象和嵌套 NDArray"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if pd.isna(obj):
            return None
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = ArxivJSONEncoder # 兼容旧版 Flask
# 对于新版 Flask (3.0+)，使用下方的 provider 方式
from flask.json.provider import DefaultJSONProvider

class CustomJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return json.dumps(obj, cls=ArxivJSONEncoder, **kwargs)

app.json = CustomJSONProvider(app)

# 全局变量存储数据，避免重复读取
DF = None

# 后台任务队列与完成结果暂存
generation_queue = queue.Queue()
completion_results = []  # 用于存储后台执行完成的信息
results_lock = threading.Lock()
current_processing_title = None  # 记录当前正在被处理的论文标题

def generation_worker():
    global current_processing_title
    while True:
        item = generation_queue.get()
        if item is None:
            break  # 停止信号
        
        try:
            title = item.get('title', '未知')
            prompt = item.get('prompt')
            current_processing_title = title  # 标记当前处理的文章
            
            print(f"\n[队列] 开始处理新任务: 《{title}》, 当前等待队列长度: {generation_queue.qsize()}")
            result_msg, is_error = process_single_prompt(prompt, title)
            with results_lock:
                completion_results.append({
                    "id": str(datetime.datetime.now().timestamp()),
                    "message": result_msg,
                    "isError": is_error
                })
            print(f"[队列] 任务处理完成。")
        except Exception as e:
            print(f"[队列] 任务处理崩溃: {e}")
        finally:
            current_processing_title = None  # 处理完毕后清空状态
            generation_queue.task_done()

# 启动后台线程处理队列
threading.Thread(target=generation_worker, daemon=True).start()

def load_data():
    global DF
    file_path = "arxiv-metadata-oai-snapshot.parquet"
    if DF is None:
        print("正在从 Parquet 文件加载数据...")
        DF = pd.read_parquet(file_path)
        print(f"成功加载 {len(DF)} 条数据")
    return DF

# HTML 模板 - 传统三件套结合 (HTML + CSS + JS)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>arXiv QueryGen - 智能论文检索问题生成器</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f4f4f9; margin: 0; padding: 20px; color: #333; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .card { border: 1px solid #eee; padding: 20px; border-radius: 8px; margin-top: 20px; line-height: 1.6; }
        .label { font-weight: bold; color: #7f8c8d; margin-top: 15px; display: block; }
        .content { margin-bottom: 10px; }
        .abstract { background: #fdf6e3; padding: 15px; border-left: 4px solid #b58900; margin-top: 10px; }
        .btn-container { text-align: center; margin-top: 30px; }
        button { background-color: #3498db; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 16px; transition: background 0.3s; }
        button:hover { background-color: #2980b9; }
        button:disabled { cursor: not-allowed; }
        #loading { display: none; color: #3498db; font-weight: bold; margin-bottom: 15px; text-align: center;}
        @keyframes spin { to { transform: rotate(360deg); } }
        /* 初始页面加载时的全屏遮罩特效 */
        #initial-loader { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255,255,255,0.95); z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
        .global-spinner { width: 50px; height: 50px; border: 5px solid #e0e0e0; border-top-color: #3498db; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 20px; }
        /* Toast 通知样式 */
        #toast { visibility: hidden; min-width: 300px; background-color: #333; color: #fff; text-align: center; border-radius: 8px; padding: 16px; position: fixed; z-index: 1000; left: 50%; bottom: 30px; transform: translateX(-50%); font-size: 15px; opacity: 0; transition: opacity 0.5s, bottom 0.5s; box-shadow: 0 4px 12px rgba(0,0,0,0.15); line-height: 1.5; white-space: pre-wrap; }
        #toast.show { visibility: visible; opacity: 1; bottom: 50px; }
        #toast.error { background-color: #e74c3c; }
        #toast.success { background-color: #2ecc71; }
        
        /* 队列状态悬浮窗 */
        #queue-status { position: fixed; top: 20px; right: 20px; background: white; padding: 10px 15px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-left: 4px solid #3498db; z-index: 1000; transition: all 0.3s; max-width: 300px; }
        #queue-status.active { border-left-color: #e67e22; background: #fffdf5;}
        #queue-header { font-weight: bold; }
        #queue-list { display: none; list-style: none; padding: 0; margin: 10px 0 0 0; font-size: 12px; max-height: 300px; overflow-y: auto; border-top: 1px dashed #ccc; padding-top: 10px;}
        #queue-list li { margin-bottom: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #555;}
    </style>
</head>
<body>
    <!-- 页面初始加载遮罩：在第一次接口返回前处于阻塞动画状态 -->
    <div id="initial-loader">
        <div class="global-spinner"></div>
        <h2 style="color: #2c3e50;">程序刚启动：正在将海量 Parquet 数据载入内存...</h2>
        <p style="color: #7f8c8d;">可能需要数十秒，请耐心等待（后续抽取将瞬间完成）⏳</p>
    </div>

    <div id="queue-status">
        <div id="queue-header">📊 队列等待数: 0</div>
        <ul id="queue-list"></ul>
    </div>
    <div class="container">
        <h1>📚 arXiv QueryGen</h1>
        <p style="text-align: center; color: #7f8c8d; margin-top: -10px; margin-bottom: 25px;">学术检索 RAG 测试集自动化构建工具</p>
        
        <div class="btn-container">
            <div id="loading">正在加载数据...</div>
            <button onclick="fetchRandomRecord()">🎲 随机挑选一个</button>
            <button id="autoGenTopBtn" onclick="autoGenerateAndSave()" style="background-color: #27ae60; margin-left: 15px;">🤖 自动生成并存入CSV</button>
        </div>

        <div id="result" class="card" style="display:none;">
            <div class="label">标题:</div>
            <div id="title" class="content" style="font-size: 1.2em; font-weight: bold;"></div>
            
            <div class="label">ID / 分类 / 日期:</div>
            <div class="content"><span id="id"></span> | <span id="categories" style="color: #e67e22;"></span> | <span id="update_date" style="color: #27ae60; font-weight: bold;"></span></div>
            
            <div id="publish_info_container" style="display:none;">
                <div class="label">发表于 (Journal Ref / DOI):</div>
                <div id="publish_info" class="content" style="color: #c0392b; font-weight: bold;"></div>
            </div>

            <div class="label">作者:</div>
            <div id="authors" class="content"></div>
            
            <div class="label">摘要:</div>
            <div id="abstract" class="abstract"></div>

            <div class="label">版本记录:</div>
            <div id="versions" class="content" style="font-size: 0.85em; color: #666;"></div>
        </div>

        <div class="card" style="margin-top: 40px; border-top: 3px solid #333;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span class="label" style="margin: 0;">📋 纯文本模式 (包含提示词)</span>
                <div>
                    <button onclick="copyPlainText()" style="padding: 5px 10px; font-size: 12px; background-color: #7f8c8d; color: white; border: none; border-radius: 4px; cursor: pointer;">一键复制</button>
                </div>
            </div>
            <textarea id="plain_text_area" readonly style="width: 100%; height: 200px; padding: 10px; border: 1px dashed #ccc; border-radius: 4px; font-family: monospace; font-size: 14px; background-color: #fafafa;"></textarea>
        </div>
    </div>
    
    <div id="toast"></div>

    <script>
        let toastTimeout = null;
        function showToast(message, isError = false) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = isError ? 'show error' : 'show success';
            
            if (toastTimeout) clearTimeout(toastTimeout);
            toastTimeout = setTimeout(() => {
                toast.className = toast.className.replace(/show\s?(error|success)?/, '');
            }, 3500);
        }

        function copyPlainText() {
            const textArea = document.getElementById('plain_text_area');
            textArea.select();
            document.execCommand('copy');
            showToast('✅ 提示词已复制到剪贴板！');
        }

        // 定时轮询后端查看是否有后台任务完成
        function pollResults() {
            fetch('/api/poll_results')
                .then(res => res.json())
                .then(resData => {
                    // 更新右上角悬浮窗的队列数量和列表
                    const queueElem = document.getElementById('queue-status');
                    const listElem = document.getElementById('queue-list');
                    
                    const processingCount = resData.current_processing ? 1 : 0;
                    const totalTasks = resData.queue_size + processingCount;
                    
                    if (totalTasks > 0) {
                        document.getElementById('queue-header').textContent = '📊 任务总数: ' + totalTasks + (resData.queue_size > 0 ? ' (排队中: ' + resData.queue_size + ')' : '');
                        queueElem.classList.add('active');
                        listElem.style.display = 'block';
                    } else {
                        document.getElementById('queue-header').textContent = '📊 队列等待数: 0';
                        queueElem.classList.remove('active');
                        listElem.style.display = 'none';
                    }
                    
                    listElem.innerHTML = '';
                    
                    // 显示正在处理的任务
                    if (resData.current_processing) {
                        const li = document.createElement('li');
                        li.style.color = '#27ae60';
                        li.style.fontWeight = 'bold';
                        li.style.fontSize = '13px';
                        li.textContent = '⚡ 处理中: ' + resData.current_processing;
                        li.title = resData.current_processing;
                        listElem.appendChild(li);
                    }
                    
                    // 显示还没开始排队等待的任务
                    resData.queued_titles.forEach(t => {
                        const li = document.createElement('li');
                        li.textContent = '⏳ 排队中: ' + t;
                        li.title = t; // hover 能看到全名
                        listElem.appendChild(li);
                    });
                    
                    // 弹出气泡通知
                    resData.results.forEach(result => {
                        const icon = result.isError ? '❌ ' : '✅ ';
                        showToast(icon + result.message, result.isError);
                    });
                })
                .catch(err => console.error("Poll Error:", err));
        }
        
        // 每 2 秒拉取一次后台结果
        setInterval(pollResults, 2000);

        async function autoGenerateAndSave() {
            const promptStr = document.getElementById('plain_text_area').value;
            const paperTitle = document.getElementById('title').textContent || '未知论文';
            
            if (!promptStr) {
                showToast('❌ 提示词为空，无法生成', true);
                return;
            }
            
            const btn = document.getElementById('autoGenTopBtn');
            const originalText = '🤖 自动生成并存入CSV';
            btn.textContent = '⏳ 请求已加入队列...';
            btn.disabled = true;
            btn.style.backgroundColor = '#95a5a6';
            
            try {
                const response = await fetch('/api/generate_and_save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: promptStr, title: paperTitle })
                });
                
                const resData = await response.json();
                
                if (response.ok) {
                    showToast('✅ ' + resData.message);
                } else {
                    let errMsg = '❌ 添加队列失败：' + (resData.error || '未知错误');
                    showToast(errMsg, true);
                }
            } catch (err) {
                showToast('❌ 请求异常：' + err, true);
            } finally {
                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.disabled = false;
                    btn.style.backgroundColor = '#27ae60';
                }, 800); // .8秒后恢复按钮，方便立即继续处理下一个
            }
        }

        const PROMPT_PREFIX = `请根据我提供的文章元数据（如标题、摘要、关键词、研究问题、方法、数据集等），**生成三个**适合作为**论文检索查询**的**中文**问题。

**核心要求：**  
- 这些问题用于**找到相似主题、方法、数据集或领域的其他文章**，而不是对当前文章内容提出一个可被直接回答的事实性问题（例如“本文的准确率是多少？”或“作者提出了什么模型？”）。  
- 问题应像用户向 AI 学术助手或 Agentic RAG 系统提问时使用的自然语言查询。  
- 问题应具备一定通用性，能够召回一批同主题、同任务或同技术路线的论文；避免使用本文特有的专有名词（如自创方法名、特定实验代号等），除非该名词已广泛代表一个研究类别，要避免检索范围过窄（如限制条件太多太具体）。  
- **生成候选问题时，可以灵活考虑是否加入时间限制（如“近三年”、“2024年以来”等），尤其对于新兴且高速迭代的领域。时间限制不作为评选优劣的依据，即带时间限制的问题与不带时间限制的问题在最终选择时具有同等竞争资格，选择最优问题完全基于其通用性、代表性以及评测检索系统的适合度。**  
- 示例（好的检索问题）：  
  - “哪些论文使用了 HotpotQA 数据集？”（数据集通用）  
  - “有哪些脑电大模型相关的工作？”（领域通用）  
  - “泡泡玛特产品的研究策略有哪些？”（主题通用）  
  - “最新的信息检索（IR）领域论文有哪些？”（可接受）  
- 反例（不合适的检索问题，因为过于具体或偏向答案）：  
  - “如何利用部分配对的多模态数据识别仅由单一模态采样的神经元类型？”（过于具体）  
  - “本文得出的结论是什么？”（答案型问题）  
  - “哪些论文使用了本文提出的 Q-flip 协议？”（特有名词，无法召回其他论文）

**输出格式：** 请直接输出这三个问题，你可以使用无序列表或逐行输出，但不要添加额外解释：
\n\n`;

        let isFirstLoad = true;

        async function fetchRandomRecord() {
            const loading = document.getElementById('loading');
            const resultDiv = document.getElementById('result');
            const initialLoader = document.getElementById('initial-loader');
            
            if (!isFirstLoad) {
                loading.style.display = 'block';
            }
            
            try {
                const response = await fetch('/api/random');
                const data = await response.json();
                
                document.getElementById('title').textContent = data.title;
                document.getElementById('id').textContent = data.id;
                document.getElementById('categories').textContent = data.categories;
                document.getElementById('update_date').textContent = data.update_date;
                document.getElementById('authors').textContent = data.authors;
                document.getElementById('abstract').textContent = data.abstract;
                
                // 设置纯文本形态的内容 (包含 Prompt 前缀)
                const plainText = `${PROMPT_PREFIX}标题：${data.title}\n日期：${data.update_date}\n作者：${data.authors}\n摘要：${data.abstract}`;
                document.getElementById('plain_text_area').value = plainText;

                // 处理发表信息 (Journal-ref 和 DOI)
                const publishContainer = document.getElementById('publish_info_container');
                const publishInfo = document.getElementById('publish_info');
                let infoParts = [];
                if (data['journal-ref']) infoParts.push(`📰 ${data['journal-ref']}`);
                if (data['doi']) infoParts.push(`🔗 DOI: ${data['doi']}`);
                
                if (infoParts.length > 0) {
                    publishInfo.textContent = infoParts.join(" | ");
                    publishContainer.style.display = 'block';
                } else {
                    publishContainer.style.display = 'none';
                }

                // 处理版本记录显示
                let versionsText = "";
                if (data.versions && Array.isArray(data.versions)) {
                    versionsText = data.versions.map(v => `${v.version}: ${v.created}`).join(" | ");
                }
                document.getElementById('versions').textContent = versionsText;
                
                resultDiv.style.display = 'block';
                
                if (isFirstLoad) {
                    showToast('✅ 数据集加载成功，现在可以开始光速挑选了！');
                }
            } catch (error) {
                alert('数据获取或解析失败，请检查后端运行状态。');
            } finally {
                loading.style.display = 'none';
                if (isFirstLoad) {
                    initialLoader.style.display = 'none';
                    isFirstLoad = false;
                }
            }
        }
        
        // 页面加载时自动获取一次
        window.onload = fetchRandomRecord;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/random')
def get_random():
    df = load_data()
    idx = random.randint(0, len(df) - 1)
    row = df.iloc[idx].to_dict()
    
    # 深度递归清理 ndarray，因为 versions 和 authors_parsed 可能是嵌套数组
    def clean_obj(obj):
        if isinstance(obj, np.ndarray):
            return [clean_obj(item) for item in obj.tolist()]
        if isinstance(obj, list):
            return [clean_obj(item) for item in obj]
        if isinstance(obj, dict):
            return {k: clean_obj(v) for k, v in obj.items()}
        if isinstance(obj, np.generic):
            return obj.item()
        if pd.isna(obj):
            return None
        return obj

    cleaned_row = clean_obj(row)
    return jsonify(cleaned_row)

@app.route('/api/generate_and_save', methods=['POST'])
def generate_and_save():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        title = data.get('title', '未知论文')
        
        if not prompt:
            return jsonify({"error": "Prompt is empty"}), 400

        # 将请求放入队列，由后台线程处理
        generation_queue.put({'prompt': prompt, 'title': title})
        q_size = generation_queue.qsize()
        
        # 弹窗提示截取一小段标题防止名称太长
        short_title = title if len(title) <= 20 else title[:20] + "..."
        return jsonify({
            "message": f"《{short_title}》已入队 (排队: {q_size})",
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_single_prompt(prompt, title):
    try:
        # 从 config.json 读取配置，如果不存在则退回使用环境变量或默认值
        api_key = os.environ.get("LLM_API_KEY", "your-api-key")
        api_base = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
        model_name = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    api_key = config_data.get("LLM_API_KEY", api_key)
                    api_base = config_data.get("LLM_API_BASE", api_base)
                    model_name = config_data.get("LLM_MODEL", model_name)
            except Exception as e:
                print(f"配置文件读取失败: {e}")
        # ================================================

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 清理 api_base 末尾可能的斜杠
        api_base = api_base.rstrip('/')
        
        # 第一阶段：生成问题
        payload_stage1 = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "你是一个专业的学术和研究助理。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        # 发送请求一给大模型 API
        response1 = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload_stage1, timeout=60)
        response1.raise_for_status()
        
        reply_json1 = response1.json()
        choices1 = reply_json1.get('choices')
        if not choices1 or not isinstance(choices1, list) or len(choices1) == 0:
            return "API 接口第1阶段返回格式异常或包含错误", True
            
        ai_content1 = choices1[0].get('message', {}).get('content')
        if not ai_content1:
            return "大模型第1阶段返回了空内容", True
            
        print("\n" + "="*40 + " API 第1阶段生成内容 " + "="*40)
        print(f"模型 ({model_name}) 返回候选问题：\n{ai_content1}")
        
        # 第二阶段：分析并选出最终结果
        prompt_stage2 = (
            "以上是你根据论文元数据生成的这三个候选检索问题。\n"
            "现在请对这三个问题进行分析，评估它们各自是否满足作为优秀检索提问的要求（例如通用性、搜索意图代表性等，时间限制不作为评价依据）。\n"
            "最后经过分析，选出一个符合要求且最终能被用来评估系统效果的问题。若都比较好，可随机选取一个。\n\n"
            "请严格按照以下 JSON 数据结构输出你的回答结果，必须是一段可解析的 JSON，不需要包含任何 Markdown block 或是其他额外解释：\n"
            "{\n"
            '  "candidate_questions": ["问题1", "问题2", "问题3"],\n'
            '  "analysis": "用一段连贯的话对这三个问题的优劣势和特点进行分析",\n'
            '  "selected_question": "最终选出的那一个问题"\n'
            "}"
        )
        
        payload_stage2 = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "你是一个专业的学术和研究助理。严格按照用户要求输出JSON格式。"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ai_content1},
                {"role": "user", "content": prompt_stage2}
            ],
            "temperature": 0.7
        }

        response2 = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload_stage2, timeout=60)
        response2.raise_for_status()
        
        reply_json2 = response2.json()
        choices2 = reply_json2.get('choices')
        if not choices2 or not isinstance(choices2, list) or len(choices2) == 0:
            return "API 接口第2阶段返回格式异常或包含错误", True
            
        ai_content2 = choices2[0].get('message', {}).get('content')
        if not ai_content2:
            return "大模型第2阶段返回了空内容", True

        print("\n" + "="*40 + " API 第2阶段生成内容 " + "="*40)
        print(f"模型 ({model_name}) 返回最终分析结果：\n{ai_content2}")
        print("="*94 + "\n")

        # 尝试清理 Markdown 标识符并解析 JSON
        try:
            json_str = re.sub(r'^```json\s*|```\s*$', '', ai_content2.strip(), flags=re.MULTILINE)
            parsed_data = json.loads(json_str)
            selected_q = parsed_data.get('selected_question')
        except json.JSONDecodeError:
            error_msg = f"AI 返回的内容不是有效的 JSON 格式\n原文: {ai_content}"
            return error_msg, True

        if not selected_q:
            error_msg = f"JSON 中未找到 'selected_question' 字段\n原文: {ai_content}"
            return error_msg, True

        # 保存到 CSV
        csv_file = "research_queries.csv"
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['timestamp', 'query']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow({
                'timestamp': timestamp,
                'query': selected_q
            })
            
        short_title = title if len(title) <= 20 else title[:20] + "..."
        return f"《{short_title}》生成并保存成功:\n{selected_q}", False
        
    except requests.exceptions.RequestException as e:
        raw_text = e.response.text if getattr(e, 'response', None) is not None else ""
        error_msg = f"《{title}》请求失败: {str(e)} | 原文: {raw_text}"
        return error_msg, True
    except Exception as e:
        error_msg = f"《{title}》发生系统异常: {str(e)}"
        return error_msg, True

@app.route('/api/poll_results')
def poll_results():
    with results_lock:
        data = list(completion_results)
        completion_results.clear()
        
    q_items = list(generation_queue.queue)
    queued_titles = [i.get('title', '未知论文') for i in q_items]
    
    return jsonify({
        "results": data,
        "queue_size": len(q_items),
        "queued_titles": queued_titles,
        "current_processing": current_processing_title
    })

if __name__ == '__main__':
    # 为了让前端页面迅速打开并显示 loading 动画，取消启动时的同步载入阻塞
    # 第一次 /api/random 调用时才会触发 Parquet 加载
    
    # 默认运行在 5000 端口，关闭 debug 模式，禁用重载功能
    app.run(debug=False, use_reloader=False, port=5000)
