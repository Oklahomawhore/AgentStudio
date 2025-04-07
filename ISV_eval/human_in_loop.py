import os
import json
import time
import threading
import webbrowser
from typing import Dict, Any, Optional, List
from collections import defaultdict
from flask import Flask, request, render_template, jsonify
import asyncio

class HumanAnnotator:
    """人类标注者类，通过网页界面让人类回答问卷"""
    
    def __init__(
        self,
        save_dir: str = "human_annotations",
        port: int = 5000,
        name="人类标注者"
    ):
        """
        初始化人类标注者类
        
        参数:
            save_dir: 保存标注结果的目录
            port: Flask 服务器的端口号
            auto_open_browser: 是否自动打开浏览器
        """
        self.save_dir = os.path.join(save_dir, name.replace(" ", "_"))
        self.port = port

        self.app = Flask(__name__)
        self.response_data = None
        self.response_ready = False
        self.pending_questions = []  # 待回答的问题队列
        self.current_question_info = None  # 当前问题信息
        self.server_running = False
        self.server_thread = None
        self.waiting_events = {}  # 用于存储各问题的等待事件
        self.server_lock = asyncio.Lock()
        self.name = name
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置Flask路由
        self._setup_routes()
        
        # 注册退出处理函数
        self._load_questions_asked()
    
    def _load_questions_asked(self):
        """
        从文件中加载已问问题
        """
        if os.path.exists(os.path.join(self.save_dir, "questions_asked.json")):
            with open(os.path.join(self.save_dir, "questions_asked.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)  # Loads as a normal dict
                self.questions_asked = defaultdict(list, data)  # Convert back to defaultdict
        else:
            self.questions_asked = defaultdict(list)

    def _save_questions_asked(self):
        """
        将已问问题保存到文件
        """
        with open(os.path.join(self.save_dir, 'questions_asked.json'), 'w', encoding='utf-8') as f:
            json.dump(self.questions_asked, f, ensure_ascii=False, indent=4)
    def __str__(self):
        return self.name
    
    def _setup_routes(self):
        """设置Flask路由"""
        
        @self.app.route('/', methods=['GET'])
        def index():
            return render_template('questionnaire.html', 
                                  questions=self.current_questions,
                                  title=self.current_title,
                                  description=self.current_description,
                                  context=self.current_context)
        
        # 在 HumanAnnotator 类中修改 submit 路由处理函数
        @self.app.route('/submit', methods=['POST'])
        def submit():
            """提交问卷结果"""
            self.response_data = request.json
            
            # 记录结果到文件
            if self.current_question_info:
                
                # 标记当前问题已回答
                self.current_question_info["answered"] = True
                self.current_question_info["answer"] = self.response_data.get("answer", "")
                
                # 通知等待此问题的任务
                question_id = self.current_question_info.get("id")
                if question_id in self.waiting_events:
                    self.waiting_events[question_id] = 1
                    
                # 清除当前问题，准备处理下一个
                self.current_question_info = None
                
                # 处理下一个问题
                if self.pending_questions:
                    self._process_next_question()
            
            # 返回响应
            if self.pending_questions:
                next_count = len(self.pending_questions)
                return jsonify({
                    "status": "success", 
                    "message": f"回答已提交，还有 {next_count} 个问题待回答",
                    "next": True
                })
            else:
                return jsonify({
                    "status": "success", 
                    "message": "所有问题已回答完毕，感谢您的参与！",
                    "next": False
                })
        
        @self.app.route('/files/<path:folder>')
        def list_files(folder):
            """列出静态文件目录内容"""
            if folder not in ['images', 'videos', 'screenshots']:
                return "Access denied", 403
                
            static_dir = os.path.join(self.app.static_folder, folder)
            if not os.path.exists(static_dir):
                return "Directory not found", 404
                
            files = os.listdir(static_dir)
            html = '<h1>Files in {}</h1><ul>'.format(folder)
            for file in files:
                file_path = '/static/{}/{}'.format(folder, file)
                html += '<li><a href="{}" target="_blank">{}</a></li>'.format(file_path, file)
            html += '</ul>'
            return html
    
    def _create_template_if_not_exists(self):
        """创建HTML模板文件，如果不存在的话"""
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        os.makedirs(template_dir, exist_ok=True)
        
        template_path = os.path.join(template_dir, 'questionnaire.html')
        if not os.path.exists(template_path):
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>电影评估问卷</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            max-width: 800px;
                            margin: 0 auto;
                            padding: 20px;
                        }
                        .question {
                            margin-bottom: 20px;
                            padding: 15px;
                            border: 1px solid #ddd;
                            border-radius: 5px;
                        }
                        .question-text {
                            font-weight: bold;
                            margin-bottom: 10px;
                        }
                        textarea {
                            width: 100%;
                            min-height: 100px;
                            padding: 10px;
                            box-sizing: border-box;
                        }
                        input[type="number"] {
                            width: 60px;
                        }
                        .rating-scale {
                            display: flex;
                            justify-content: space-between;
                            margin: 10px 0;
                        }
                        .rating-scale span {
                            font-size: 0.8em;
                            color: #666;
                        }
                        button {
                            background-color: #4CAF50;
                            color: white;
                            padding: 10px 15px;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 16px;
                        }
                        button:hover {
                            background-color: #45a049;
                        }
                        .header {
                            margin-bottom: 30px;
                        }
                        .title {
                            font-size: 24px;
                            font-weight: bold;
                        }
                        .description {
                            margin-top: 10px;
                            color: #666;
                        }
                        .context {
                            margin-top: 20px;
                            padding: 15px;
                            background-color: #f9f9f9;
                            border-left: 4px solid #4CAF50;
                        }
                        .context-title {
                            font-weight: bold;
                            margin-bottom: 10px;
                        }
                        .progress {
                            margin-bottom: 20px;
                            background-color: #f3f3f3;
                            border-radius: 3px;
                            height: 20px;
                        }
                        .progress-bar {
                            height: 100%;
                            background-color: #4CAF50;
                            border-radius: 3px;
                            text-align: center;
                            color: white;
                            font-size: 12px;
                            line-height: 20px;
                        }
                        .thumbnail {
                            max-width: 100%;
                            max-height: 300px;
                            margin: 10px 0;
                        }
                        .video-container {
                            position: relative;
                            width: 100%;
                            margin: 15px 0;
                        }
                        video {
                            width: 100%;
                            max-height: 450px;
                        }
                        .video-controls {
                            margin-top: 5px;
                            display: flex;
                            gap: 10px;
                            align-items: center;
                        }
                        .video-controls button {
                            padding: 5px 10px;
                            font-size: 14px;
                        }
                        .image-gallery {
                            display: flex;
                            flex-wrap: wrap;
                            gap: 10px;
                            margin: 10px 0;
                        }
                        .image-gallery img {
                            max-width: 200px;
                            max-height: 150px;
                            object-fit: cover;
                            border: 1px solid #ddd;
                            border-radius: 4px;
                            cursor: pointer;
                        }
                        .modal {
                            display: none;
                            position: fixed;
                            z-index: 999;
                            left: 0;
                            top: 0;
                            width: 100%;
                            height: 100%;
                            background-color: rgba(0,0,0,0.9);
                        }
                        .modal-content {
                            margin: auto;
                            display: block;
                            max-width: 90%;
                            max-height: 90%;
                        }
                        .close {
                            position: absolute;
                            top: 15px;
                            right: 35px;
                            color: #f1f1f1;
                            font-size: 40px;
                            font-weight: bold;
                            cursor: pointer;
                        }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <div class="title">{{ title }}</div>
                        <div class="description">{{ description }}</div>
                    </div>
                    
                    {% if context %}
                    <div class="context">
                        <div class="context-title">评估背景信息：</div>
                        
                        {% if context.video %}
                        <div class="video-container">
                            <p><strong>视频文件:</strong> {{ context.video }}</p>
                            <video id="videoPlayer" controls>
                                <source src="/static/videos/{{ context.video_filename }}" type="video/mp4">
                                您的浏览器不支持HTML5视频播放器
                            </video>
                            <div class="video-controls">
                                <button onclick="document.getElementById('videoPlayer').playbackRate = 0.5">0.5x</button>
                                <button onclick="document.getElementById('videoPlayer').playbackRate = 1.0">1.0x</button>
                                <button onclick="document.getElementById('videoPlayer').playbackRate = 1.5">1.5x</button>
                                <button onclick="document.getElementById('videoPlayer').playbackRate = 2.0">2.0x</button>
                                <span id="currentTime">00:00</span> / <span id="duration">00:00</span>
                            </div>
                        </div>
                        {% endif %}
                        
                        {% if context.text %}
                        <div>
                            <p><strong>场景描述:</strong> {{ context.text }}</p>
                        </div>
                        {% endif %}
                        
                        {% if context.image %}
                        <div>
                            <p><strong>场景图像:</strong></p>
                            <div class="image-gallery">
                                {% if context.image is string %}
                                    <img src="/static/images/{{ context.image }}" class="thumbnail" alt="场景图像" onclick="openModal('/static/images/{{ context.image }}')">
                                {% else %}
                                    {% for img in context.image %}
                                    <img src="/static/images/{{ img }}" class="thumbnail" alt="场景图像 {{ loop.index }}" onclick="openModal('/static/images/{{ img }}')">
                                    {% endfor %}
                                {% endif %}
                            </div>
                        </div>
                        {% endif %}
                    </div>
                    {% endif %}
                    
                    <!-- 图像查看模态框 -->
                    <div id="imageModal" class="modal">
                        <span class="close" onclick="closeModal()">&times;</span>
                        <img class="modal-content" id="modalImage">
                    </div>
                    
                    <form id="questionnaireForm">
                        {% for question in questions %}
                        <div class="question">
                            <div class="question-text">{{ question.text }}</div>
                            
                            {% if question.type == 'rating' %}
                                <div class="rating-scale">
                                    <span>{{ question.min_label }} ({{ question.min }})</span>
                                    <span>{{ question.max_label }} ({{ question.max }})</span>
                                </div>
                                <input type="number" name="{{ question.id }}" min="{{ question.min }}" max="{{ question.max }}" required>
                            {% elif question.type == 'text' %}
                                <textarea name="{{ question.id }}" placeholder="{{ question.placeholder }}" required></textarea>
                            {% elif question.type == 'options' %}
                                {% for option in question.options %}
                                <div>
                                    <input type="radio" id="{{ question.id }}_{{ loop.index }}" name="{{ question.id }}" value="{{ option }}" required>
                                    <label for="{{ question.id }}_{{ loop.index }}">{{ option }}</label>
                                </div>
                                {% endfor %}
                            {% elif question.type == 'yes_no' %}
                                <div>
                                    <input type="radio" id="{{ question.id }}_yes" name="{{ question.id }}" value="是" required>
                                    <label for="{{ question.id }}_yes">是</label>
                                </div>
                                <div>
                                    <input type="radio" id="{{ question.id }}_no" name="{{ question.id }}" value="否" required>
                                    <label for="{{ question.id }}_no">否</label>
                                </div>
                            {% elif question.type == 'scoring_1_10' %}
                                <div class="rating-scale">
                                    <span>最低 (1)</span>
                                    <span>最高 (10)</span>
                                </div>
                                <input type="number" name="{{ question.id }}" min="1" max="10" required>
                            {% endif %}
                        </div>
                        {% endfor %}
                        
                        <button type="submit">提交回答</button>
                    </form>

                    <script>
                        // 视频播放时间更新
                        if (document.getElementById('videoPlayer')) {
                            const videoPlayer = document.getElementById('videoPlayer');
                            const currentTimeDisplay = document.getElementById('currentTime');
                            const durationDisplay = document.getElementById('duration');
                            
                            // 格式化时间为 MM:SS 格式
                            function formatTime(seconds) {
                                const minutes = Math.floor(seconds / 60);
                                const secs = Math.floor(seconds % 60);
                                return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                            }
                            
                            videoPlayer.addEventListener('loadedmetadata', function() {
                                durationDisplay.textContent = formatTime(videoPlayer.duration);
                            });
                            
                            videoPlayer.addEventListener('timeupdate', function() {
                                currentTimeDisplay.textContent = formatTime(videoPlayer.currentTime);
                            });
                        }
                        
                        // 图像模态框功能
                        function openModal(imageSrc) {
                            const modal = document.getElementById('imageModal');
                            const modalImg = document.getElementById('modalImage');
                            modal.style.display = 'flex';
                            modalImg.src = imageSrc;
                        }
                        
                        function closeModal() {
                            document.getElementById('imageModal').style.display = 'none';
                        }
                        
                        // 点击模态框背景关闭
                        window.onclick = function(event) {
                            const modal = document.getElementById('imageModal');
                            if (event.target == modal) {
                                modal.style.display = 'none';
                            }
                        }
                        
                        // 问卷提交处理
                        document.getElementById('questionnaireForm').addEventListener('submit', function(e) {
                            e.preventDefault();
                            
                            // 收集表单数据
                            const formData = new FormData(this);
                            const jsonData = {};
                            
                            for (const [key, value] of formData.entries()) {
                                jsonData[key] = value;
                            }
                            
                            // 组织答案格式
                            // 对于yes_no问题，直接用第一个字段的值
                            // 对于scoring_1_10问题，直接用第一个字段的值作为分数
                            const firstKey = Object.keys(jsonData)[0];
                            const firstValue = jsonData[firstKey];
                            
                            // 根据问题类型设置回答格式
                            jsonData.answer = firstValue;
                            jsonData.score = firstValue;
                            
                            // 发送数据到服务器
                            fetch('/submit', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify(jsonData),
                            })
                            .then(response => response.json())
                            .then(data => {
                                if (data.status === 'success') {
                                    alert(data.message);
                                    if (data.next) {
                                        // 如果还有下一个问题，刷新页面
                                        window.location.reload();
                                    } else {
                                        // 如果没有更多问题，显示完成消息
                                        document.body.innerHTML = '<div style="text-align:center; margin-top:100px;"><h2>所有问题已回答完毕</h2><p>感谢您的参与，您可以关闭此窗口。</p></div>';
                                    }
                                }
                            })
                            .catch(error => {
                                alert('提交出错: ' + error);
                            });
                        });
                    </script>
                </body>
                </html>
                ''')
    
    async def do_questionare(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        实现与MovieReviewCommission兼容的do_questionare接口
        
        参数:
            question: 问题文本
            context: 问题上下文，如视频路径、场景描述等
            
        返回:
            人类标注者的回答
        """
        # 检查是否已有相同问题的回答
        if question in self.questions_asked:
            return self.questions_asked[question]
        # 将问题添加到队列
        question_info = {
            "question": question,
            "context": context or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "answered": False
        }
        
        # 添加到待回答队列
        self.pending_questions.append(question_info)
        
        # 如果服务器未运行，启动服务器
        await self._start_server()

        if self.pending_questions:
            self._process_next_question()
        
        # 等待此问题被回答
        result = await self._wait_for_answer(question_info)
        return result
            
    async def _wait_for_answer(self, question_info):
        """等待问题被回答，使用事件机制"""
        # 创建唯一标识，用作问题ID
        question_id = str(hash(question_info["question"]) + hash(str(question_info["context"])))
        question_info["id"] = question_id
        
        # 创建一个事件，用于等待此问题被回答
        
        self.waiting_events[question_id] = 0
        
        while not self.waiting_events[question_id]:
            await asyncio.sleep(0.1)  # 等待回答
        # 删除事件
        del self.waiting_events[question_id]
        self.questions_asked[question_info["question"]].append({self.name: question_info["answer"]})
        self._save_questions_asked()
        return question_info.get("answer", "")
        
    def _process_next_question(self):
        """处理队列中的下一个问题"""
        if not self.pending_questions:
            return
            
        # 取出第一个问题
        self.current_question_info = self.pending_questions.pop(0)
        question_text = self.current_question_info["question"]
        context = self.current_question_info["context"]
        
        # 准备问题显示
        if "yes or no" in question_text:
            # 是否问题
            clean_question = question_text.replace("anwer this question with yes or no:", "").strip()
            self.current_questions = [{
                "id": "answer",
                "type": "yes_no",
                "text": clean_question
            }]
            self.current_title = "是/否问题"
            
        elif "1-10" in question_text:
            # 评分问题
            clean_question = question_text.replace("anwer this question with a score of 1-10:", "").strip()
            self.current_questions = [{
                "id": "score",
                "type": "scoring_1_10",
                "text": clean_question
            }]
            self.current_title = "评分问题"
            
        else:
            # 其他问题
            self.current_questions = [{
                "id": "answer",
                "type": "text",
                "text": question_text,
                "placeholder": "请输入您的回答..."
            }]
            self.current_title = "问题"
            
        # 设置描述和上下文
        if context and 'video' in context:
            video_name = os.path.basename(context['video']).split('.')[0]
            self.current_description = f"请对视频《{video_name}》进行评估"
        else:
            self.current_description = "请回答以下问题"
            
        # 处理上下文文件
        self._prepare_context_files(context)
        
        self.current_context = context
    
    def _prepare_context_files(self, context):
        """准备上下文文件（视频和图像）供网页访问"""
        if not context:
            return
            
        # 处理视频文件
        if 'video' in context and os.path.exists(context['video']):
            # 确保静态视频目录存在
            static_videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/videos')
            os.makedirs(static_videos_dir, exist_ok=True)
            
            # 为视频创建符号链接或复制
            video_filename = os.path.basename(context['video'])
            static_video_path = os.path.join(static_videos_dir, video_filename)
            
            try:
                # 先尝试创建符号链接（更高效）
                if os.path.exists(static_video_path):
                    os.remove(static_video_path)
                os.symlink(os.path.abspath(context['video']), static_video_path)
                context['video_filename'] = video_filename
            except Exception as e:
                print(f"创建视频符号链接失败，尝试复制：{e}")
                try:
                    # 如果符号链接失败，尝试复制文件
                    import shutil
                    shutil.copy(context['video'], static_video_path)
                    context['video_filename'] = video_filename
                except Exception as e2:
                    print(f"复制视频文件失败：{e2}")
        
        # 处理图像文件
        if 'image' in context:
            # 确保静态图像目录存在
            static_images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/images')
            os.makedirs(static_images_dir, exist_ok=True)
            
            if isinstance(context['image'], str) and os.path.exists(context['image']):
                # 单个图像
                image_filename = os.path.basename(context['image'])
                static_image_path = os.path.join(static_images_dir, image_filename)
                
                try:
                    # 复制或创建链接
                    if not os.path.exists(static_image_path):
                        import shutil
                        shutil.copy(context['image'], static_image_path)
                    context['image'] = image_filename
                except Exception as e:
                    print(f"处理图像文件失败：{e}")
                    
            elif isinstance(context['image'], list):
                # 多个图像
                processed_images = []
                for img_path in context['image']:
                    if os.path.exists(img_path):
                        image_filename = os.path.basename(img_path)
                        static_image_path = os.path.join(static_images_dir, image_filename)
                        
                        try:
                            # 复制或创建链接
                            if not os.path.exists(static_image_path):
                                import shutil
                                shutil.copy(img_path, static_image_path)
                            processed_images.append(image_filename)
                        except Exception as e:
                            print(f"处理图像文件失败：{e}")
                
                context['image'] = processed_images
        
    async def _start_server(self):
        """启动Flask服务器"""
        
        async with self.server_lock:
            # 确保模板存在
            if not self.server_running:
                self._create_template_if_not_exists()
                
                # 在新线程中启动Flask服务器
                self.server_thread = threading.Thread(target=self._run_server)
                self.server_thread.daemon = True
                self.server_thread.start()

                # 标记服务器为运行状态
                self.server_running = True

                print(f"标注问卷已启动，请在浏览器中访问 http://localhost:{self.port} 进行回答")
                print(f"待回答问题数量: {len(self.pending_questions)}")

    def _run_server(self):
        """在线程中运行Flask服务器"""
        # 添加静态文件支持
        static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        os.makedirs(static_folder, exist_ok=True)
        
        # 创建必要的子目录
        for subdir in ['images', 'videos', 'screenshots']:
            os.makedirs(os.path.join(static_folder, subdir), exist_ok=True)
        
        # 设置Flask的静态文件夹
        self.app.static_folder = static_folder
        self.app.static_url_path = '/static'
        
        # 启动服务器
        self.app.run(host='0.0.0.0', port=self.port, use_reloader=False)
    
        
    def get_questionnaire_results(self) -> Dict[str, Any]:
        """获取所有问卷结果，与 MovieReviewCommission 类似"""
        return self.questions_asked

# 下面是测试代码
if __name__ == '__main__':
    annotator = HumanAnnotator(save_dir="human_annotations")
    
    # 测试简单问题
    import asyncio
    
    async def test():
        answer = await annotator.do_questionare(
            question="anwer this question with yes or no: 这部电影的情节是否连贯?",
            context={"video": "sample.mp4"}
        )
        print(f"问题回答: {answer}")
        
        score = await annotator.do_questionare(
            question="anwer this question with a score of 1-10: 这部电影的视觉效果质量如何?",
            context={"video": "sample.mp4"}
        )
        print(f"评分结果: {score}")
    
    asyncio.run(test())