with open("ISV_train/resp.txt", "r") as f:
    resp = f.read()
import json
import sys
sys.path.append("/data/wangshu/wangshu_code/ISG")
sys.path.append("/data/wangshu/wangshu_code/ISG/ISG_agent")
from ISG_agent.PlanningAgentV2 import extract_json_from_response

extracted_json = extract_json_from_response(resp)
print(extracted_json)
data= json.loads(extracted_json)
print(data)