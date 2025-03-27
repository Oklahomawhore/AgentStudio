from typing import List, Dict, Any, Optional, Union
import asyncio
import json
from agents import BaseAgent,CriticAgent,CulturalExpertAgent, AudienceAgent

class MovieReviewCommission:
    """电影评审委员会系统"""
    
    def __init__(self, agents: List[BaseAgent]):
        """
        初始化电影评审委员会
        
        参数:
            agents: 评审委员会成员，包括不同类型的智能体
        """
        self.agents = agents
        self.discussions = []
        self.votes = {}
        self.final_decision = {}
    
    async def evaluate_film(self, film_title: str, film_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        对电影进行全面评估
        
        参数:
            film_title: 电影标题
            film_info: 电影相关信息
            
        返回:
            综合评估结果
        """
        evaluations = {}
        
        # 收集每个智能体的评估
        for agent in self.agents:
            if isinstance(agent, CriticAgent):
                result = await agent.evaluate_film(film_title, film_info)
                evaluations[f"{agent.characteristics.name}_critic"] = result
            
            elif isinstance(agent, CulturalExpertAgent):
                result = await agent.analyze_cultural_elements(film_title, film_info)
                evaluations[f"{agent.characteristics.name}_cultural"] = result
            
            elif isinstance(agent, AudienceAgent):
                result = await agent.rate_audience_experience(film_title, film_info)
                evaluations[f"{agent.characteristics.name}_audience"] = result
            
            else:
                # 对于其他类型的智能体，使用通用方法
                prompt = f"请评估电影《{film_title}》并给出你的专业意见。电影信息：{json.dumps(film_info, ensure_ascii=False)}"
                response = await agent._respond_to_user(prompt)
                evaluations[agent.characteristics.name] = response
        
        return evaluations
    
    async def conduct_discussion(self, film_title: str, discussion_topics: List[str]) -> List[Dict[str, Any]]:
        """
        组织智能体进行讨论
        
        参数:
            film_title: 电影标题
            discussion_topics: 讨论话题列表
            
        返回:
            讨论记录
        """
        discussion_records = []
        
        for topic in discussion_topics:
            topic_discussion = {
                "topic": topic,
                "responses": []
            }
            
            # 每个智能体对当前话题发表意见
            for agent in self.agents:
                response = await agent._respond_to_user(
                    f"关于电影《{film_title}》的以下话题，请发表你的观点：{topic}"
                )
                
                topic_discussion["responses"].append({
                    "agent": agent.characteristics.name,
                    "response": response
                })
            
            # 智能体之间进行交叉讨论
            for i, agent1 in enumerate(self.agents):
                for j, agent2 in enumerate(self.agents):
                    if i != j:
                        # 智能体1对智能体2的观点做出回应
                        agent2_opinion = topic_discussion["responses"][j]["response"]
                        response = await agent1._respond_to_user(
                            f"请回应{agent2.characteristics.name}关于《{film_title}》{topic}的以下观点：\n\n{agent2_opinion}"
                        )
                        
                        topic_discussion["responses"].append({
                            "agent": agent1.characteristics.name,
                            "response_to": agent2.characteristics.name,
                            "response": response
                        })
            
            discussion_records.append(topic_discussion)
            self.discussions.append(topic_discussion)
        
        return discussion_records
    
    async def vote_on_film(self, film_title: str, voting_criteria: List[str]) -> Dict[str, Any]:
        """
        对电影进行投票
        
        参数:
            film_title: 电影标题
            voting_criteria: 投票标准列表，如"艺术价值"、"商业价值"等
            
        返回:
            投票结果
        """
        votes = {criterion: {} for criterion in voting_criteria}
        
        for agent in self.agents:
            # 构建投票提示词
            vote_prompt = f"""
            请对电影《{film_title}》按照以下标准进行评分，每项1-10分：
            {', '.join(voting_criteria)}
            
            同时，请简要说明你的评分理由。
            请以JSON格式返回你的投票结果。
            """
            
            # 获取投票
            response = await agent._respond_to_user(vote_prompt)
            
            try:
                # 尝试解析JSON格式的投票结果
                vote_result = json.loads(response)
            except:
                # 如果解析失败，记录原始响应
                vote_result = {"raw_response": response}
            
            # 记录投票
            for criterion in voting_criteria:
                if criterion in vote_result:
                    votes[criterion][agent.characteristics.name] = {
                        "score": vote_result[criterion],
                        "reason": vote_result.get(f"{criterion}_reason", "未提供理由")
                    }
        
        # 计算每个标准的平均分
        for criterion in voting_criteria:
            scores = [v["score"] for v in votes[criterion].values() if isinstance(v.get("score"), (int, float))]
            if scores:
                votes[criterion]["average"] = sum(scores) / len(scores)
            else:
                votes[criterion]["average"] = "无有效评分"
        
        self.votes = votes
        return votes
    
    async def generate_final_report(self, film_title: str, film_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成最终评审报告
        
        参数:
            film_title: 电影标题
            film_info: 电影相关信息
            
        返回:
            最终报告
        """
        # 准备报告内容
        report_content = {
            "film_title": film_title,
            "film_info": film_info,
            "evaluations_summary": {},
            "discussions_summary": {},
            "voting_results": self.votes,
            "final_rating": None,
            "recommendation": None
        }
        
        # 选择一个智能体生成总结报告
        summarizer_agent = self.agents[0]  # 默认使用第一个智能体作为总结者
        
        # 寻找专业影评人作为总结者
        for agent in self.agents:
            if isinstance(agent, CriticAgent):
                summarizer_agent = agent
                break
        
        # 构建总结提示词
        if self.votes:
            voting_summary = json.dumps(self.votes, ensure_ascii=False, indent=2)
        else:
            voting_summary = "未进行投票"
        
        summary_prompt = f"""
        请对电影《{film_title}》的评审过程进行总结，并生成最终评审报告。
        
        电影信息：
        {json.dumps(film_info, ensure_ascii=False, indent=2)}
        
        投票结果：
        {voting_summary}
        
        请从以下方面总结评审结果：
        1. 电影艺术价值评估
        2. 文化意义评估
        3. 观众接受度评估
        4. 最终评分(1-100分)
        5. 是否推荐上映/发行的建议
        6. 简短的总体评价
        
        请以JSON格式返回最终报告。
        """
        
        # 获取总结报告
        response = await summarizer_agent._respond_to_user(summary_prompt)
        
        try:
            final_report = json.loads(response)
        except:
            final_report = {"raw_report": response}
        
        # 更新最终报告
        report_content.update({
            "final_rating": final_report.get("final_rating") or final_report.get("score"),
            "recommendation": final_report.get("recommendation"),
            "summary": final_report
        })
        
        self.final_decision = report_content
        return report_content

# 使用示例
async def main():
    # 创建评审委员会成员
    critic = CriticAgent(
        specialty="艺术电影",
        critic_style="严苛"
    )
    
    cultural_expert = CulturalExpertAgent(
        cultural_background="中国文化",
        expertise_areas=["电影中的文化符号", "东西方文化比较"]
    )
    
    audience_rep = AudienceAgent(
        demographics={"age_group": "18-25岁", "occupation": "大学生", "education": "本科在读", "region": "二线城市"}
    )
    
    # 创建评审委员会
    commission = MovieReviewCommission([critic, cultural_expert, audience_rep])
    
    # 电影信息
    film_title = "《流浪地球》"
    film_info = {
        "director": "郭帆",
        "cast": ["吴京", "李光洁", "屈楚萧"],
        "genres": ["科幻", "灾难"],
        "year": 2019,
        "countries": ["中国"],
        "synopsis": "太阳即将毁灭，人类在地球表面建造了巨大的推进器，寻求将地球推离太阳系，寻找新家园。"
    }
    
    # 评估电影
    evaluations = await commission.evaluate_film(film_title, film_info)
    print("评估完成")
    
    # 讨论电影
    discussion_topics = [
        "电影对中国传统文化价值观的表达",
        "科学设定的合理性与艺术表现",
        "电影的国际市场表现与跨文化传播"
    ]
    discussions = await commission.conduct_discussion(film_title, discussion_topics)
    print("讨论完成")
    
    # 投票
    voting_criteria = ["艺术价值", "科学准确性", "文化意义", "商业潜力", "国际影响力"]
    votes = await commission.vote_on_film(film_title, voting_criteria)
    print("投票完成")
    
    # 生成最终报告
    final_report = await commission.generate_final_report(film_title, film_info)
    print("最终报告生成完成")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())