import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional
from pprint import pprint
import warnings
import traceback

# import langchain
# langchain.debug = True
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import re
from langchain_core.messages import BaseMessage
import sys
import os
from time import time, sleep
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import (
    get_caption_prompt, 
    get_object_extraction_prompt, 
    get_tool_caption_prompt,
    get_tool_caption_benchmark_prompt, 
    get_obj_retrieval_prompt, 
    get_tool_call_example_1,
    get_tool_call_example_2)
from language_planner.llm_backend.tools import AgentToolbox
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.utils import secret_from_env
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import MessagesState
from mistralai import Mistral

from llm_backend.langgraph_agent import build_graph, build_actor_critic_graph, ActorCriticState, CriticStructuredOutput
from llm_backend.enums import SystemMode, LanguageModel, NavQueryRunMode, ObjectQueryType

class LLMQueryHandler:

    def __init__(
            self, 
            model = LanguageModel.MISTRAL, 
            run_mode = NavQueryRunMode.DEFAULT, 
            system_mode = SystemMode.LIVE_NAVIGATION, 
            **kwargs):

        self.model = model
        self.run_mode = run_mode
        self.system_mode = system_mode
        self.rate_limiter = InMemoryRateLimiter(
                requests_per_second=0.18, #1.2, # 5 seconds between requests
                check_every_n_seconds=0.1,
            )
        
        
        # Initialize base LLM
        if model == LanguageModel.GPT4:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    base_url="https://cmu.litellm.ai",
                    model_name="gpt-4",
                    **kwargs
                )
            except Exception as e:
                print(f"Error initializing OpenAI model: {e}")
        elif model == LanguageModel.GEMINI:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    **kwargs
                )
            except Exception as e:
                print(f"Error initializing Google Generative AI model: {e}")
        elif model == LanguageModel.MISTRAL:
            from langchain_mistralai import ChatMistralAI
            self.llm = ChatMistralAI(
                model="mistral-large-latest",
                temperature=0.0,
                random_seed=0,
                rate_limiter = self.rate_limiter,
                max_retries=10
            )
        elif model == LanguageModel.LLAMA or model == LanguageModel.R1_QWEN2:
            # install ollama here: https://github.com/ollama/ollama
            # Run in a separate terminal as `ollama run llama3.1:8b` or `ollama run deepseek-r1:7b`
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=model.value,
                temperature=0.0,
                num_predict=5096,
                **kwargs
            )
        elif model == LanguageModel.GPT4O:
            #set "OPENAI_API_KEY" env variable
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="gpt-4o"
            )
        elif model == LanguageModel.GPT4O_MINI:
            self.llm = ChatOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="gpt-4o-mini"
            )
        else:
            raise NotImplementedError(f"Model {model} not supported")
        
        if self.run_mode != NavQueryRunMode.DEFAULT:
            self.toolbox = AgentToolbox(self.system_mode)
        
        self._setup_chains()


    def _set_up_thinking(self):
        # return a Runnablelambda that will remove everything within <think> tag from the input if using R1 models, otherwise, return a RunnablePassthrough
        def remove_think_tag(input):
            res = re.sub(r"<think>(.*?)</think>", "", input.content, flags=re.DOTALL)
            input.content = res
            return input
        
        if self.model == LanguageModel.R1_QWEN2:
            return RunnableLambda(remove_think_tag)
        else:
            return RunnablePassthrough()


    def _setup_chains(self):
        think = self._set_up_thinking()

        if self.run_mode == NavQueryRunMode.USE_TOOL_NOT_GRAPH:
            from langchain.agents import AgentExecutor
            from langchain.agents import create_tool_calling_agent
            nav_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "{prompt}"),
                    ("human", "User Input: \n {query}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )
            agent = create_tool_calling_agent(self.llm, self.toolbox.tools, prompt=nav_prompt)
            self.nav_chain = (AgentExecutor(agent=agent, tools=self.toolbox.tools, verbose=True, return_intermediate_steps=True)
                            | RunnableLambda(lambda x: x['output'])
                            | StrOutputParser())
        elif self.run_mode == NavQueryRunMode.USE_TOOL_USE_GRAPH:
            self.nav_chain = build_graph(
                tools=self.toolbox.tools,
                llm_with_tools=self.llm.bind_tools(self.toolbox.tools, tool_choice="any"),
            )
        elif self.run_mode == NavQueryRunMode.USE_TOOL_ACTOR_CRITIC_GRAPH:
            self.nav_chain = build_actor_critic_graph(
                tools=self.toolbox.tools,
                actor_with_tools=self.llm.bind_tools(self.toolbox.tools),
                critic_llm=self.llm.with_structured_output(CriticStructuredOutput)
            )
        elif self.run_mode == NavQueryRunMode.DEFAULT:
            nav_prompt = ChatPromptTemplate.from_messages([
                ("system", "{prompt}"),
                ("human", "User Input: \n {query}")
            ])
            self.nav_chain = (
                nav_prompt 
                | self.llm 
                | think
                | StrOutputParser()
            )

        # Object extraction chain
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", "{prompt}"),
            ("human", "{query}")
        ])


        self.extract_chain = (
            extract_prompt 
            | self.llm 
            | think
            | StrOutputParser()
        )


    def generate_query(
            self,
            environment_name: str,
            grid_map_shape: tuple,
            robot_coords: np.ndarray,
            objects: List[Dict],
            input_query: str,
            object_dict: dict = None,
            map_pcl: np.ndarray = None,
            freespace_pcl: np.ndarray = None
        ) -> str:
        """Generate navigation query response"""
        if self.run_mode == NavQueryRunMode.DEFAULT:
            prompt = get_caption_prompt(environment_name, grid_map_shape, robot_coords, objects)
        else:
            from langchain_core.tools.render import render_text_description_and_args
            tool_descriptions = render_text_description_and_args(self.toolbox.tools)
            if self.run_mode == NavQueryRunMode.USE_TOOL_NOT_GRAPH:
                prompt = get_caption_prompt(environment_name, grid_map_shape, robot_coords, objects)
                prompt += f'''You have access to the following tools, you should use them as much as possible during intermediate steps to help you generate the code, do not attempt to figure out yourself if you can use a tool to help you. You should only output content when you are sure and confident about the task, use tools if you are not. You should always use the notepad first to write down your reasoning and thoughts before dealing with a task.
                        {tool_descriptions}
                        '''
            else:
                if self.system_mode == SystemMode.LIVE_NAVIGATION:
                    prompt = get_tool_caption_prompt(environment_name, grid_map_shape, robot_coords, objects, tool_descriptions)
                else:
                    prompt = get_tool_caption_benchmark_prompt(environment_name, grid_map_shape, robot_coords, objects, tool_descriptions)

                
        if self.run_mode in [NavQueryRunMode.DEFAULT, NavQueryRunMode.USE_TOOL_NOT_GRAPH]:
            try:
                response = self.nav_chain.invoke({"prompt": prompt, "query": input_query})
                return response
            except Exception as e:
                warnings.warn(f"Error generating navigation query: {e}")
                return f"Error generating navigation query: {e}"
        else:
            warning = ""
            
            # --- 修改开始：将消息对象转换成纯文本字符串的辅助逻辑 ---
            def format_examples_to_text(msgs):
                text = ""
                for m in msgs:
                    if isinstance(m, HumanMessage):
                        text += f"\n[User Input]: {m.content}\n"
                    elif isinstance(m, AIMessage):
                        if m.tool_calls:
                            for tc in m.tool_calls:
                                text += f"[AI Action]: Calling tool '{tc['name']}' with args {tc['args']}\n"
                        if m.content:
                            text += f"[AI Thought]: {m.content}\n"
                    elif isinstance(m, ToolMessage):
                        text += f"[Tool Output]: {m.content}\n"
                return text

            example1_msgs = get_tool_call_example_1(self.system_mode == SystemMode.BENCHMARK)
            example_text = "\n=== EXAMPLE 1 ===\n" + format_examples_to_text(example1_msgs)
            
            if self.system_mode == SystemMode.LIVE_NAVIGATION:
                example2_msgs = get_tool_call_example_2()
                example_text += "\n=== EXAMPLE 2 ===\n" + format_examples_to_text(example2_msgs)

            # 重新构建 nav_prompt：让示例变成 SystemMessage 的一部分，不再是独立对象
            nav_prompt = [
                SystemMessage(content=prompt + "\n\nFollow this reasoning and tool usage format:\n" + example_text),
                HumanMessage(content="End of examples. Now please solve the following task FRESH based on the current object list.\n\nObject List: \n" + str(objects) + "\n\nUser Input: \n" + input_query)
            ]
            # 后面保持不变
            if self.run_mode == NavQueryRunMode.USE_TOOL_USE_GRAPH:

                if map_pcl is not None:
                    self.toolbox.pcl = map_pcl
                if freespace_pcl is not None:
                    self.toolbox.pcl = freespace_pcl
                if object_dict is not None:
                    self.toolbox.set_object_dict(object_dict)

                state = MessagesState(messages=nav_prompt)
            elif self.run_mode == NavQueryRunMode.USE_TOOL_ACTOR_CRITIC_GRAPH:
                if map_pcl is not None:
                    self.toolbox.pcl = map_pcl
                if freespace_pcl is not None:
                    self.toolbox.pcl = freespace_pcl
                if object_dict is not None:
                    self.toolbox.set_object_dict(object_dict)
                state = ActorCriticState(messages=nav_prompt, objects=objects, critic_approval=False)
            
            try:
                state = self.nav_chain.invoke(state, config={"recursion_limit": 50})
            except Exception as e:
                warnings.warn(f"Error generating navigation query: {e}")
                traceback.print_exc()
                # warning = f"Error generating navigation query: {e}"
                # warning += traceback.format_exc()
                # print("============ERROR STATE==============")
                # print(state)
                # print("=====================================")
            # Parse messages
            # print("============STATE==============")
            # print(state)
            # print("=====================================")
            steps = []
            for message in state["messages"]:
                # 如果消息内容太长（通常是那个 Object List），就不要打印它的内容
                content_preview = (message.content[:100] + "...") if len(message.content) > 100 else message.content
                
                if isinstance(message, AIMessage) and len(message.tool_calls) > 0:
                    for tool_call in message.tool_calls:
                        steps.append(f"AI Action: Calling {tool_call['name']}")
                elif isinstance(message, AIMessage):
                    steps.append(f"AI Thought: {content_preview}")
                elif isinstance(message, ToolMessage):
                    steps.append(f"Tool Result: {content_preview}")
                elif isinstance(message, HumanMessage):
                    # 这里的 HumanMessage 往往包含了几千字的物体列表，我们只打印前 50 个字
                    steps.append(f"Human: {content_preview}")

            traces = "\n".join(steps)
            cmds = self.toolbox.cmd # [('go_near', (1,)), ('go_between', (2, 3))]
            thoughts = self.toolbox.notes
            self.toolbox.clear()
            code_lines = []
            for cmd, args in cmds:
                args_str = ','.join(str(arg) for arg in args)
                code_lines.append(f"    {cmd}({args_str})")
            response = "Code:\ndef go():\n" + "\n".join(code_lines)
            print("-----------------")
            print(warning + "Traces:\n" + traces + "\n" + "Reasoning: \n" + "\n".join(thoughts) + "\n" + response)
            print("-----------------")
            
            return warning + "Traces:\n" + traces + "\n" + "Reasoning: \n" + "\n".join(thoughts) + "\n" + response


    def extract_objects(self, input_query: str) -> List[str]:
        """Extract objects from natural language query"""

        result = None
        tries = 3
        while tries > 0:
            try:
                result = self.extract_chain.invoke({"prompt": get_object_extraction_prompt(), "query": input_query})
                return eval(result)
            except Exception as e:
                warnings.warn(f"Error extracting objects for statement '{input_query}': {e}")
                print("Retrying...")
                pprint(result)
                sleep(1)
                tries -= 1
        
        return []


    def filter_objects(self, query_list, semantic_dict):
        prompt = get_obj_retrieval_prompt()
        prompt_obj_dict = {obj_id:obj_info["name"] for obj_id, obj_info in semantic_dict.items()}

        prompt_suffix = f'''Targets={query_list} \n Scene objects={prompt_obj_dict} \n Output:\n
        '''
        prompt += prompt_suffix
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        tries = 3
        while tries > 0:
            try:
                chat_response = client.chat.complete(
                    model = "mistral-large-latest",
                    messages = [
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt_suffix,
                        }
                    ]
                )
                tries = 0
                print("model output", chat_response)
            except Exception as e:
                print("Retrying")
                tries -= 1
                sleep(2)
        
        response = chat_response.choices[0].message.content
        try:
            return eval(response)
        except Exception as e:
            pprint(response)
            print("Mistral response is not evaluable!")
            start = '['
            end=']'
            re_parsed = response[response.rfind(start)+len(start):response.rfind(end)]
            if re_parsed is not None:
                re_parsed = '[' + re_parsed + ']'
            try: 
                print("parsed str", re_parsed)
                return eval(re_parsed)
            except Exception as e:
                print(e)
                pprint(response)
                print("parsed string cannot be evaluated!")
                return []
                

    def __call__(self, *args, **kwargs):
        """Allow direct calling of generate_query"""
        return self.generate_query(*args, **kwargs)
    