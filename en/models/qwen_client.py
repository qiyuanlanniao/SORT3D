# iot@iot:~/hm/ros2_ws/src/en/models/qwen_client.py

from openai import OpenAI
import time

class QwenVLMClient:
    """
    Qwen2.5-VL 客户端
    通过 OpenAI 兼容接口连接本地 vLLM 服务器
    用于获取推理结果并记录 Token 消耗
    """
    def __init__(self, api_key="EMPTY", base_url="http://localhost:8000/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    def request(self, prompt):
        """
        发送请求并返回结构化数据
        """
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # 实验中使用 0.0 确保结果可复现
                max_tokens=5096    # 导航指令通常不需要太长回复
            )
            
            duration = time.time() - start_time
            
            # 提取文本内容
            content = response.choices[0].message.content
            
            # 提取关键的 Token 统计信息信息信息
            # vLLM 服务器会在 usage 字段中返回精准计数
            usage = response.usage
            
            return {
                "success": True,
                "answer": content,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "latency": round(duration, 2)
            }

        except Exception as e:
            print(f"ERROR: Failed to connect to vLLM server: {e}")
            return {
                "success": False,
                "error": str(e),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency": 0
            }

# ---------------------------------------------------------
# 单元测试 (需先启动 vLLM 服务器)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 简单的连接测试
    client = QwenVLMClient()
    test_prompt = "Say 'Hello World' and count to 3."
    
    print(f"--- Connecting to {client.model_name} ---")
    res = client.request(test_prompt)
    
    if res["success"]:
        print(f"Response: {res['answer']}")
        print(f"Tokens Used: {res['total_tokens']} (In: {res['input_tokens']}, Out: {res['output_tokens']})")
        print(f"Latency: {res['latency']}s")
    else:
        print("Check if your vLLM server is running at http://localhost:8000/v1")
