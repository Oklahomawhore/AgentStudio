import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI

@dataclass
class Memory:
    """智能体的记忆结构"""
    conversations: List[Dict[str, Any]] = field(default_factory=list)
    knowledge: Dict[str, Any] = field(default_factory=dict)
    
    def add_conversation(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """添加一条对话记录到记忆中"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.conversations.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
    
    def get_recent_conversations(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近的n条对话"""
        return self.conversations[-n:] if len(self.conversations) >= n else self.conversations
    
    def add_knowledge(self, key: str, value: Any):
        """向知识库添加信息"""
        self.knowledge[key] = value
    
    def get_knowledge(self, key: str) -> Any:
        """从知识库获取信息"""
        return self.knowledge.get(key)


@dataclass
class Characteristics:
    """智能体的内部特征"""
    name: str = "Agent"
    personality: str = "Helpful and informative"
    goals: List[str] = field(default_factory=lambda: ["Provide accurate information", "Be helpful"])
    expertise: List[str] = field(default_factory=lambda: ["General knowledge"])
    
    def get_prompt_description(self) -> str:
        """获取描述智能体特征的提示词"""
        return f"""
        You are {self.name}, an AI agent with the following characteristics:
        - Personality: {self.personality}
        - Goals: {', '.join(self.goals)}
        - Expertise: {', '.join(self.expertise)}
        """


class BaseAgent:
    """基础智能体类"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o", 
        characteristics: Optional[Characteristics] = None,
        memory: Optional[Memory] = None
    ):
        """
        初始化基础智能体
        
        参数:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            model: 使用的OpenAI模型名称
            characteristics: 智能体特征，如果为None则使用默认特征
            memory: 智能体记忆，如果为None则创建新的记忆
        """
        # 初始化OpenAI客户端
        self.api_key = api_key or os.environ.get("KLING_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OpenAI API密钥未提供，请通过参数传入或设置OPENAI_API_KEY环境变量")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model
        
        # 初始化特征和记忆
        self.characteristics = characteristics or Characteristics()
        self.memory = memory or Memory()
        
    def _generate_system_prompt(self) -> str:
        """生成系统提示词"""
        return self.characteristics.get_prompt_description()
    
    def _prepare_messages(self, recent_message_count: int = 10) -> List[Dict[str, str]]:
        """准备发送给API的消息列表"""
        messages = [{"role": "system", "content": self._generate_system_prompt()}]
        
        # 添加最近的对话历史
        recent_conversations = self.memory.get_recent_conversations(recent_message_count)
        for conv in recent_conversations:
            messages.append({"role": conv["role"], "content": conv["content"]})
            
        return messages
        
    async def talk(self, message: str, other_agent: Optional['BaseAgent'] = None) -> str:
        """
        与智能体交谈或让该智能体与另一个智能体交谈
        
        参数:
            message: 发送给智能体的消息
            other_agent: 可选，另一个要交谈的智能体
            
        返回:
            智能体的回复
        """
        if other_agent:
            # 如果提供了另一个智能体，则让当前智能体与其交谈
            return await self._talk_to_agent(message, other_agent)
        else:
            # 否则，回应用户的消息
            return await self._respond_to_user(message)
    
    async def _respond_to_user(self, message: str) -> str:
        """回应用户的消息"""
        # 记录用户消息
        self.memory.add_conversation("user", message)
        
        # 准备发送给API的消息
        messages = self._prepare_messages()
        
        # 调用API获取回复
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        reply = response.choices[0].message.content
        
        # 记录智能体回复
        self.memory.add_conversation("assistant", reply)
        
        return reply
    
    async def _talk_to_agent(self, message: str, other_agent: 'BaseAgent') -> str:
        """与另一个智能体交谈"""
        # 记录发送给其他智能体的消息
        self.memory.add_conversation("user", f"[To {other_agent.characteristics.name}]: {message}")
        
        # 获取其他智能体的回复
        reply = await other_agent._respond_to_user(message)
        
        # 记录其他智能体的回复
        self.memory.add_conversation("assistant", f"[From {other_agent.characteristics.name}]: {reply}")
        
        return reply
    
    def update_characteristics(self, **kwargs):
        """更新智能体特征"""
        for key, value in kwargs.items():
            if hasattr(self.characteristics, key):
                setattr(self.characteristics, key, value)
    
    def clear_memory(self):
        """清除智能体的记忆"""
        self.memory = Memory()

class AudienceAgent(BaseAgent):
    """代表普通观众视角的智能体"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        demographics: Dict[str, Any] = None,
        viewing_preferences: List[str] = None,
        viewing_history: List[str] = None,
        **kwargs
    ):
        """
        初始化观众代表智能体
        
        参数:
            api_key: OpenAI API密钥
            model: 使用的OpenAI模型
            demographics: 观众人口统计信息
            viewing_preferences: 观影偏好
            viewing_history: 观影历史
            **kwargs: 传递给BaseAgent的其他参数
        """
        # 默认人口统计信息
        if demographics is None:
            demographics = {
                "age_group": "25-35岁",
                "occupation": "普通职场人士",
                "education": "大学本科",
                "region": "一线城市"
            }
        
        # 默认观影偏好
        if viewing_preferences is None:
            viewing_preferences = ["动作", "科幻", "喜剧", "悬疑"]
            
        # 默认观影历史
        if viewing_history is None:
            viewing_history = ["复仇者联盟系列", "星际穿越", "少年的你", "寄生虫"]
        
        # 设置观众代表特征
        audience_characteristics = Characteristics(
            name=f"{demographics['age_group']}观众代表",
            personality="真诚、直率且注重情感体验",
            goals=[
                "从普通观众角度评价电影的娱乐性", 
                "关注电影的情感共鸣和共情度", 
                "评估电影的吸引力和可看性"
            ],
            expertise=[
                "观众体验",
                "情感反应",
                "大众文化鉴赏",
                "娱乐性评价"
            ]
        )
        
        # 初始化基础智能体
        super().__init__(
            api_key=api_key, 
            model=model,
            characteristics=kwargs.get('characteristics', audience_characteristics),
            memory=kwargs.get('memory', None)
        )
        
        # 添加观众代表特有的属性
        self.demographics = demographics
        self.viewing_preferences = viewing_preferences
        self.viewing_history = viewing_history
        
        # 添加观众信息到记忆
        self.memory.add_knowledge("demographics", self.demographics)
        self.memory.add_knowledge("viewing_preferences", self.viewing_preferences)
        self.memory.add_knowledge("viewing_history", self.viewing_history)
    
    def _generate_system_prompt(self) -> str:
        """重写系统提示词生成方法，加入普通观众视角"""
        base_prompt = super()._generate_system_prompt()
        
        audience_prompt = f"""
        你代表一名{self.demographics['age_group']}的{self.demographics['occupation']}观众，教育程度为{self.demographics['education']}，生活在{self.demographics['region']}。
        
        你喜欢观看{', '.join(self.viewing_preferences)}类型的电影。
        近期看过的电影包括：{', '.join(self.viewing_history)}。
        
        在评价电影时，你主要关注：
        1. 娱乐性和观赏体验
        2. 故事是否引人入胜
        3. 角色是否可信和有共鸣
        4. 是否值得票价和时间
        5. 是否会推荐给朋友
        
        请用日常口语表达你的真实感受，就像和朋友聊天一样自然。
        """
        
        return base_prompt + audience_prompt
    
    async def rate_audience_experience(self, film_title: str, film_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        从观众角度评价电影体验
        
        参数:
            film_title: 电影标题
            film_info: 电影相关信息
            
        返回:
            观众体验评分和评价
        """
        # 构建观众体验评价提示词
        audience_prompt = f"""
        假设你刚看完电影《{film_title}》，请从普通观众的角度给出你的真实感受。电影信息如下：
        - 类型: {', '.join(film_info.get('genres', ['未知']))}
        - 主演: {', '.join(film_info.get('cast', ['未知']))}
        - 片长: {film_info.get('duration', '未知')}分钟
        - 简介: {film_info.get('synopsis', '无简介')}
        
        请从以下方面给出评价：
        1. 娱乐性 (1-10分)：电影是否有趣、刺激或感人
        2. 故事吸引力 (1-10分)：故事是否引人入胜
        3. 角色共鸣 (1-10分)：是否能与角色产生共鸣
        4. 值回票价 (1-10分)：观影体验是否超过票价成本
        5. 推荐意愿 (1-10分)：是否会推荐给朋友观看
        
        同时，请分享：
        1. 最喜欢的场景或部分
        2. 最不满意的地方
        3. 这部电影让你有什么感受
        4. 是否会再看一次
        
        请以JSON格式返回你的评价。
        """
        
        # 获取观众评价
        response = await self._respond_to_user(audience_prompt)
        
        # 记录评价到记忆
        self.memory.add_knowledge(f"audience_rating_{film_title}", response)
        
        try:
            import json
            result = json.loads(response)
            return result
        except:
            return {"raw_rating": response}
    
    async def compare_with_similar_films(self, film_title: str, similar_films: List[str]) -> str:
        """与类似电影进行比较"""
        prompt = f"请将《{film_title}》与以下类似电影进行比较：{', '.join(similar_films)}。从观众体验角度，这部电影相比之下有何优缺点？"
        return await self._respond_to_user(prompt)
    

class CriticAgent(BaseAgent):
    """专业电影评论家智能体"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        specialty: str = "综合影评",
        critic_style: str = "平衡",
        publications: List[str] = None,
        **kwargs
    ):
        """
        初始化专业影评人智能体
        
        参数:
            api_key: OpenAI API密钥
            model: 使用的OpenAI模型
            specialty: 评论专长领域，如"艺术电影"、"商业大片"、"纪录片"等
            critic_style: 评论风格，如"严苛"、"宽容"、"平衡"等
            publications: 曾发表评论的出版物列表
            **kwargs: 传递给BaseAgent的其他参数
        """
        # 设置默认的评论家特征
        critic_characteristics = Characteristics(
            name=f"{specialty}评论家",
            personality=f"{critic_style}、专业且富有洞察力的电影评论者",
            goals=[
                "提供专业的电影艺术评析", 
                "考量电影的艺术价值和社会影响", 
                "给出公平客观的评价"
            ],
            expertise=[
                f"{specialty}领域的电影评论",
                "电影艺术理论",
                "叙事结构分析",
                "视觉语言解读"
            ]
        )
        
        # 初始化基础智能体
        super().__init__(
            api_key=api_key, 
            model=model,
            characteristics=kwargs.get('characteristics', critic_characteristics),
            memory=kwargs.get('memory', None)
        )
        
        # 添加专业影评人特有的属性
        self.specialty = specialty
        self.critic_style = critic_style
        self.publications = publications or ["电影艺术", "看电影", "电影评论"]
        
        # 添加专业知识到记忆
        self.memory.add_knowledge("specialty", self.specialty)
        self.memory.add_knowledge("critic_style", self.critic_style)
        self.memory.add_knowledge("publications", self.publications)
    
    def _generate_system_prompt(self) -> str:
        """重写系统提示词生成方法，加入更多专业评论背景"""
        base_prompt = super()._generate_system_prompt()
        
        critic_specific_prompt = f"""
        作为一名专攻{self.specialty}的电影评论家，你使用{self.critic_style}的评论风格。
        你的评论曾发表在{', '.join(self.publications)}等媒体平台。
        
        在评估电影时，你应当关注：
        1. 导演的艺术表达和视觉语言
        2. 剧本的结构和故事叙事
        3. 演员表演和角色塑造
        4. 摄影、剪辑、音效等技术元素
        5. 电影的主题深度和社会意义
        
        请用专业但平易近人的语言表达你的观点，避免过于学术化的术语。
        """
        
        return base_prompt + critic_specific_prompt
    
    async def evaluate_film(self, film_title: str, film_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        对电影进行专业评估
        
        参数:
            film_title: 电影标题
            film_info: 电影相关信息，包括导演、主演、剧情等
            
        返回:
            评估结果字典，包含分数和评语
        """
        # 构建电影评估提示词
        evaluation_prompt = f"""
        请对《{film_title}》进行专业评估。电影信息如下：
        - 导演: {film_info.get('director', '未知')}
        - 主演: {', '.join(film_info.get('cast', ['未知']))}
        - 类型: {', '.join(film_info.get('genres', ['未知']))}
        - 上映年份: {film_info.get('year', '未知')}
        - 简介: {film_info.get('synopsis', '无简介')}
        
        请从以下方面进行评价：
        1. 导演手法 (20分)
        2. 剧本/故事 (20分)
        3. 演员表演 (20分)
        4. 技术元素 (20分)
        5. 主题深度 (20分)
        
        针对每个方面给出1-20分的评分，并提供简短的评语。
        然后给出总评分(1-100)和总体评价。
        请以JSON格式返回评价结果。
        """
        
        # 获取评估结果
        response = await self._respond_to_user(evaluation_prompt)
        
        # 记录评估到记忆
        self.memory.add_knowledge(f"evaluation_{film_title}", response)
        
        try:
            # 这里假设返回的是格式良好的JSON字符串
            # 在实际应用中，可能需要更复杂的解析逻辑
            import json
            result = json.loads(response)
            return result
        except:
            # 如果解析失败，直接返回原始响应
            return {"raw_evaluation": response}
        

class CulturalExpertAgent(BaseAgent):
    """文化背景专家智能体，关注电影中的文化表达与影响"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        cultural_background: str = "中国文化",
        expertise_areas: List[str] = None,
        academic_background: str = "文化研究学者",
        **kwargs
    ):
        """
        初始化文化背景专家智能体
        
        参数:
            api_key: OpenAI API密钥
            model: 使用的OpenAI模型
            cultural_background: 专家的文化背景
            expertise_areas: 专家研究领域
            academic_background: 学术背景
            **kwargs: 传递给BaseAgent的其他参数
        """
        # 设置默认的专家特征
        expert_characteristics = Characteristics(
            name=f"{cultural_background}专家",
            personality="严谨、包容且富有文化视角",
            goals=[
                "评估电影的文化准确性与敏感度", 
                "分析电影中的文化符号和隐喻", 
                "评价电影对特定文化的描绘是否尊重和真实"
            ],
            expertise=[
                f"{cultural_background}研究",
                "跨文化比较",
                "文化符号学",
                "电影中的文化表达"
            ]
        )
        
        # 默认研究领域
        if expertise_areas is None:
            expertise_areas = [
                "传统文化在现代电影中的表达", 
                "文化刻板印象分析", 
                "电影中的文化元素真实性"
            ]
        
        # 初始化基础智能体
        super().__init__(
            api_key=api_key, 
            model=model,
            characteristics=kwargs.get('characteristics', expert_characteristics),
            memory=kwargs.get('memory', None)
        )
        
        # 添加文化专家特有的属性
        self.cultural_background = cultural_background
        self.expertise_areas = expertise_areas
        self.academic_background = academic_background
        
        # 添加专业知识到记忆
        self.memory.add_knowledge("cultural_background", self.cultural_background)
        self.memory.add_knowledge("expertise_areas", self.expertise_areas)
        self.memory.add_knowledge("academic_background", self.academic_background)
    
    def _generate_system_prompt(self) -> str:
        """重写系统提示词生成方法，加入更多文化专家背景"""
        base_prompt = super()._generate_system_prompt()
        
        cultural_expert_prompt = f"""
        你是一位专注于{self.cultural_background}的{self.academic_background}，研究领域包括{', '.join(self.expertise_areas)}。
        
        在评估电影时，你应特别关注：
        1. 电影对文化元素的准确描绘
        2. 文化刻板印象和偏见的存在
        3. 跨文化交流的准确性和敏感度
        4. 电影中隐含的文化价值观和世界观
        5. 电影对文化多样性的尊重程度
        
        请用专业但易于理解的语言表达你的观点，避免使用过于学术化的术语，同时保持文化敏感性。
        """
        
        return base_prompt + cultural_expert_prompt
    
    async def analyze_cultural_elements(self, film_title: str, film_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析电影中的文化元素
        
        参数:
            film_title: 电影标题
            film_info: 电影相关信息
            
        返回:
            文化分析结果
        """
        # 构建文化分析提示词
        culture_prompt = f"""
        请对《{film_title}》中的文化元素进行专业分析。电影信息如下：
        - 导演: {film_info.get('director', '未知')}
        - 制作国家/地区: {', '.join(film_info.get('countries', ['未知']))}
        - 文化背景: {film_info.get('cultural_setting', '未指定')}
        - 上映年份: {film_info.get('year', '未知')}
        - 简介: {film_info.get('synopsis', '无简介')}
        
        请从以下方面进行文化分析：
        1. 文化准确性 (1-20分)：电影对所描绘文化的准确性
        2. 文化敏感度 (1-20分)：电影对文化差异的尊重程度
        3. 刻板印象 (1-20分)：电影是否避免或强化文化刻板印象
        4. 文化多样性 (1-20分)：电影对文化多样性的展现
        5. 文化深度 (1-20分)：电影对文化内涵的探索深度
        
        针对每个方面给出评分和分析，并提供总体评价(1-100分)。
        请以JSON格式返回分析结果。
        """
        
        # 获取分析结果
        response = await self._respond_to_user(culture_prompt)
        
        # 记录分析到记忆
        self.memory.add_knowledge(f"cultural_analysis_{film_title}", response)
        
        try:
            import json
            result = json.loads(response)
            return result
        except:
            return {"raw_analysis": response}
    
    async def evaluate_cultural_impact(self, film_title: str, target_culture: str) -> str:
        """评估电影对特定文化的影响"""
        prompt = f"请评估电影《{film_title}》对{target_culture}的潜在影响。考虑电影中的文化表达如何被该文化群体感知，以及可能产生的社会影响。"
        return await self._respond_to_user(prompt)