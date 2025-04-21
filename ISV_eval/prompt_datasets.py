from torch.utils.data import Dataset
import json
import os

class VideoStorytellingNovel(Dataset):
    """
    Dataset class for video storytelling with novel prompts.
    This dataset loads Chinese stories from the provided JSON file and returns the prompt text.
    """
    
    def __init__(self, json_path, transform=None):
        """
        Initialize the VideoStorytellingNovel dataset.
        
        Args:
            json_path: Path to the JSON file containing story data
            transform: Optional transform to be applied on the data
        """
        self.transform = transform
        self.data = []
        
        # Load data from JSON file
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)
        
        # Process each story entry
        for story in stories:
            story_id = story.get("id", "")
            category = story.get("Category", "")
            
            # Extract query content
            query = story.get("Query", [])
            query_content = ""
            for q in query:
                if q.get("type") == "text":
                    query_content = q.get("content", "")
                    break
            
            if query_content:
                self.data.append({
                    "id": story_id,
                    "category": category,
                    "prompt": query_content
                })
    
    def __len__(self):
        """Return the number of stories in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a story item by index.
        
        Args:
            idx: Index of the story to retrieve
            
        Returns:
            Dictionary containing story ID, category, and prompt text
        """
        item = self.data[idx]
        
        # Apply transforms if specified
        if self.transform:
            item = self.transform(item)
            
        return item
    
    def get_prompt(self, idx):
        """
        Get only the prompt text for a story by index.
        
        Args:
            idx: Index of the story to retrieve prompt from
            
        Returns:
            Prompt text string
        """
        return self.data[idx]["prompt"]


class QuestionType:
    FILL_IN_THE_BLANK = "fill_in_the_blank"
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"


class Question:
    """基础问题类，包含问题和答案"""
    def __init__(self, question_text, answer):
        self.question_text = question_text
        self.answer = answer

    def to_dict(self):
        """返回问题的字典表示"""
        return {
            "question": self.question_text,
            "answer": self.answer
        }


class FillInBlankQuestion(Question):
    """填空题"""
    def __init__(self, question_text, answer):
        super().__init__(question_text, answer)


class YesNoQuestion(Question):
    """是非题"""
    def __init__(self, question_text, answer):
        super().__init__(question_text, answer)
        # 确保答案是 Yes 或 No
        if answer not in ["Yes", "No", "yes", "no"]:
            raise ValueError("YesNoQuestion answer must be 'Yes' or 'No'")


class MultipleChoiceQuestion(Question):
    """多选题"""
    def __init__(self, question_text, options, answer):
        super().__init__(question_text, answer)
        if isinstance(options, list):
            self.options = options
        elif isinstance(options, dict):
            self.options = list(options.values())

        # 确保答案在选项中
        if answer not in ['A', 'B', 'C', 'D']:
            if str(answer[0]).upper()  in ['A', 'B', 'C', 'D']:
                self.answer  = str(answer[0]).upper()
            elif answer in self.options:
                self.answer = ['A', 'B', 'C', 'D'][self.options.index(answer)]
            else:
                raise ValueError("Answer must be one of the options: 'A', 'B', 'C', 'D'")
        else:
            self.answer = answer
    def to_dict(self):
        """返回多选题的字典表示，包含选项"""
        return {
            "question": self.question_text,
            "options": self.options,
            "answer": self.answer
        }


class StoryQuestions:
    """故事问题集合类"""
    def __init__(self, story_id, fill_in_questions=None, yes_no_questions=None, multiple_choice_questions=None):
        self.story_id = story_id
        self.fill_in_questions = fill_in_questions or []
        self.yes_no_questions = yes_no_questions or []
        self.multiple_choice_questions = multiple_choice_questions or []
    
    def add_fill_in_question(self, question):
        """添加填空题"""
        if isinstance(question, FillInBlankQuestion):
            self.fill_in_questions.append(question)
        else:
            raise TypeError("Expected FillInBlankQuestion object")
    
    def add_yes_no_question(self, question):
        """添加是非题"""
        if isinstance(question, YesNoQuestion):
            self.yes_no_questions.append(question)
        else:
            raise TypeError("Expected YesNoQuestion object")
    
    def add_multiple_choice_question(self, question):
        """添加多选题"""
        if isinstance(question, MultipleChoiceQuestion):
            self.multiple_choice_questions.append(question)
        else:
            raise TypeError("Expected MultipleChoiceQuestion object")
    
    def get_all_questions(self):
        """获取所有问题"""
        return {
            QuestionType.FILL_IN_THE_BLANK: self.fill_in_questions,
            QuestionType.YES_NO: self.yes_no_questions,
            QuestionType.MULTIPLE_CHOICE: self.multiple_choice_questions
        }


class StoryQuestionDataset:
    """故事问题数据集类"""
    def __init__(self, questions_dir):
        """
        初始化故事问题数据集
        
        Args:
            questions_dir: 包含问题JSON文件的目录路径
        """
        self.questions_dir = questions_dir
        self.questions = {}  # 按故事ID存储问题
        
        # 加载所有问题文件
        self._load_questions()
    
    def _load_questions(self):
        """从指定目录加载所有问题文件"""
        if not os.path.exists(self.questions_dir):
            raise FileNotFoundError(f"Questions directory not found at: {self.questions_dir}")
        
        # 扫描目录中的所有JSON文件
        for filename in os.listdir(self.questions_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.questions_dir, filename)
                story_id = os.path.splitext(filename)[0]
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    question_data = json.load(f)
                    
                    # 创建故事问题集合
                    story_questions = StoryQuestions(story_id)
                    
                    # 处理填空题
                    if QuestionType.FILL_IN_THE_BLANK in question_data:
                        for q_item in question_data[QuestionType.FILL_IN_THE_BLANK]:
                            question = FillInBlankQuestion(q_item["question"], q_item["answer"])
                            story_questions.add_fill_in_question(question)
                    
                    # 处理是非题
                    if QuestionType.YES_NO in question_data:
                        for q_item in question_data[QuestionType.YES_NO]:
                            question = YesNoQuestion(q_item["question"], q_item["answer"])
                            story_questions.add_yes_no_question(question)
                    
                    # 处理多选题
                    if QuestionType.MULTIPLE_CHOICE in question_data:
                        for q_item in question_data[QuestionType.MULTIPLE_CHOICE]:
                            question = MultipleChoiceQuestion(
                                q_item["question"],
                                q_item["options"],
                                q_item["answer"]
                            )
                            story_questions.add_multiple_choice_question(question)
                    
                    # 添加到数据集
                    self.questions[story_id] = story_questions
    
    def get_questions_by_id(self, story_id):
        """通过故事ID获取问题集合"""
        if story_id in self.questions:
            return self.questions[story_id]
        return None
    
    def get_questions_by_type(self, story_id, question_type):
        """通过故事ID和问题类型获取问题列表"""
        if story_id not in self.questions:
            return []
        
        if question_type == QuestionType.FILL_IN_THE_BLANK:
            return self.questions[story_id].fill_in_questions
        elif question_type == QuestionType.YES_NO:
            return self.questions[story_id].yes_no_questions
        elif question_type == QuestionType.MULTIPLE_CHOICE:
            return self.questions[story_id].multiple_choice_questions
        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    def has_questions_for_story(self, story_id):
        """检查是否有针对特定故事的问题"""
        return story_id in self.questions


class EnhancedVideoStorytellingDataset(VideoStorytellingNovel):
    """增强型视频讲故事数据集，带有问题功能"""
    
    def __init__(self, json_path, questions_dir, transform=None):
        """
        初始化增强型数据集
        
        Args:
            json_path: 故事JSON文件路径
            questions_dir: 问题文件目录路径
            transform: 可选的数据转换函数
        """
        super().__init__(json_path, transform)
        self.questions = StoryQuestionDataset(questions_dir)
    
    def get_story_questions(self, idx):
        """获取故事的所有问题"""
        story_id = self.data[idx]["id"]
        return self.questions.get_questions_by_id(story_id)
    
    def get_story_questions_by_type(self, idx, question_type):
        """获取特定类型的故事问题"""
        story_id = self.data[idx]["id"]
        return self.questions.get_questions_by_type(story_id, question_type)
    
    def has_questions(self, idx):
        """检查故事是否有问题"""
        story_id = self.data[idx]["id"]
        return self.questions.has_questions_for_story(story_id)
    def __len__(self):
        return super().__len__()