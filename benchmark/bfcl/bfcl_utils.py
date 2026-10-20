"""BFCL 任务评测工具函数。"""

import json
from pathlib import Path
from typing import Dict, List, Any

from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
)
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.model_handler.utils import (
    convert_to_tool,
    default_decode_execute_prompting,
    func_doc_language_specific_pre_processing,
)


def load_test_case(data_path: str, test_id: str | None) -> Dict[str, Any]:
    """
    按 id 加载测试用例。
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"BFCL data file '{data_path}' not found")

    if test_id is None:
        raise ValueError("task_id is required")

    with open(data_path, "r", encoding="utf-8") as f:
        if str(test_id).isdigit():  # pylint: disable=R1720
            idx = int(test_id)
            for line_no, line in enumerate(f):
                if line_no == idx:
                    return json.loads(line)
            raise ValueError(f"Test case index {idx} not found in {data_path}")
        else:
            for line in f:
                data = json.loads(line)
                if data.get("id") == test_id:
                    return data
            raise ValueError(f"Test case id '{test_id}' not found in {data_path}")


def handle_user_turn(
    test_entry: Dict[str, Any],
    current_turn: int,
) -> Dict[str, Any]:
    """
    处理用户轮次，返回 test_entry["question"] 中对应的内容。
    对于非首轮，会处理用户查询和工具列表。

    参数：
        test_entry: 包含对话数据的测试条目
        current_turn: 当前轮次编号

    返回：
        包含下一条用户消息与工具列表的响应
    """
    try:
        current_turn_message = []
        tools = compile_tools(test_entry)
        questions = test_entry.get("question", [])
        holdout_function = test_entry.get("holdout_function", {})

        if str(current_turn) in holdout_function:
            test_entry["function"].extend(holdout_function[str(current_turn)])
            tools = compile_tools(test_entry)
            assert len(questions[current_turn]) == 0, "Holdout turn should not have user message."
            current_turn_message = [
                {
                    "role": "user",
                    "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                },
            ]
            return create_user_response(current_turn_message, tools)
        if current_turn >= len(questions):
            return create_completion_response()

        current_turn_message = questions[current_turn]

        return create_user_response(current_turn_message, tools)

    except Exception as e:
        return create_error_response(f"Failed to process user message: {str(e)}")


def handle_tool_calls(  # pylint: disable=W0613
    tool_calls: List[Dict[str, Any]],
    decoded_calls: list[str],
    test_entry: Dict[str, Any],
    current_turn: int,
    execution_model_name: str = "env_handler",
) -> Dict[str, Any]:
    """
    处理 assistant 发起的工具调用。

    参数：
        tool_calls: OpenAI 格式的工具调用列表
        decoded_calls: 解码后的函数调用列表
        test_entry: 包含环境数据的测试条目
        current_turn: 当前轮次编号
        execution_model_name: BFCL 环境实例命名空间

    返回：
        包含工具执行结果的响应
    """
    execution_results, _ = execute_multi_turn_func_call(
        func_call_list=decoded_calls,
        initial_config=test_entry["initial_config"],
        involved_classes=test_entry["involved_classes"],
        model_name=execution_model_name,
        test_entry_id=test_entry["id"],
        long_context=("long_context" in test_entry["id"] or "composite" in test_entry["id"]),
        is_evaL_run=False,
    )
    # print('handler_tool_calls 中的 execution_results:', execution_results)

    return create_tool_response(tool_calls, execution_results)


def compile_tools(test_entry: dict) -> list:
    """
    将函数信息编译为 tools 格式。

    参数：
        test_entry: 包含函数定义的测试条目

    返回：
        OpenAI 格式的工具列表
    """
    functions: list = test_entry["function"]
    test_category: str = test_entry["id"].rsplit("_", 1)[0]

    functions = func_doc_language_specific_pre_processing(functions, test_category)
    tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OpenAI_Completions)

    return tools


def create_tool_response(
    tool_calls: List[Dict[str, Any]],
    execution_results: List[str],
) -> Dict[str, Any]:
    """
    为工具调用创建响应。

    参数：
        tool_calls: 工具调用列表
        execution_results: 执行结果列表

    返回：
        包含工具执行结果的响应
    """
    tool_messages = []
    for i, (tool_call, result) in enumerate(zip(tool_calls, execution_results)):
        tool_messages.append(
            {
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.get("id", f"call_{i}"),
            },
        )

    return {"messages": tool_messages}


def create_user_response(
    question_turn: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    创建包含用户消息的响应。

    参数：
        question_turn: 当前轮次的消息列表
        tools: 可用工具列表

    返回：
        包含用户消息与工具列表的响应
    """
    user_content = ""
    for msg in question_turn:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    return {"messages": [{"role": "user", "content": user_content}], "tools": tools}


def create_completion_response() -> Dict[str, Any]:
    """
    创建表示对话完成的响应。

    返回：
        带有完成消息的响应
    """
    return {"messages": [{"role": "env", "content": "[CONVERSATION_COMPLETED]"}]}


def create_error_response(error_message: str) -> Dict[str, Any]:
    """
    为错误场景创建响应。

    参数：
        error_message: 要包含的错误信息

    返回：
        包含错误信息的响应
    """
    return {"messages": [{"role": "env", "content": f"[ERROR] {error_message}"}]}


def decode_execute(result):
    """
    解码执行结果，以兼容评测框架。

    参数：
        result: 待解码结果

    返回：
        解码后的函数调用列表
    """
    return default_decode_execute_prompting(result)


def extract_single_turn_response(messages: List[Dict[str, Any]]) -> str:
    """
    从对话消息中提取单轮响应。

    参数：
        messages: 对话消息列表

    返回：
        响应的字符串表示
    """
    for message in reversed(messages):
        if message["role"] == "assistant":
            if "tool_calls" in message and message["tool_calls"]:
                formatted_calls = []
                for tool_call in message["tool_calls"]:
                    formatted_call = format_single_tool_call_for_eval(
                        tool_call,
                    )
                    if formatted_call:
                        formatted_calls.append(formatted_call)
                return "\n".join(formatted_calls) if formatted_calls else ""
            elif message.get("content"):
                return message["content"]

    return ""


def extract_multi_turn_responses(
    messages: List[Dict[str, Any]],
) -> List[List[str]]:
    """
    从对话消息中提取多轮响应。

    参数：
        messages: 对话消息列表

    返回：
        轮次列表，每一轮是函数调用字符串列表
    """
    # 防守性检查：确保messages是列表
    if not isinstance(messages, list):
        print(f"WARNING: extract_multi_turn_responses received non-list type: {type(messages)}")
        if isinstance(messages, str):
            print(f"  String content: {messages[:200]}")
        return [[]]
    
    turns_data = []
    current_turn_responses = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content", "")
        if role in {"user", "env"} and isinstance(content, str) and (
            content == "[CONVERSATION_COMPLETED]" or content.startswith("[ERROR]")
        ):
            if current_turn_responses is not None:
                turns_data.append(current_turn_responses)
            current_turn_responses = None
            continue
        if role == "user":
            if current_turn_responses is not None:
                turns_data.append(current_turn_responses)
            # An empty response is still a real task turn. Dropping it shifts
            # later calls and makes the checker report a false turn-count error.
            current_turn_responses = []
        elif role == "assistant" and current_turn_responses is not None:
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
                    continue
                formatted_call = format_single_tool_call_for_eval(tool_call)
                if formatted_call:
                    current_turn_responses.append(formatted_call)
    if current_turn_responses is not None:
        turns_data.append(current_turn_responses)
    return turns_data


def format_single_tool_call_for_eval(tool_call: Dict[str, Any]) -> str:
    """
    将单个工具调用格式化为评测使用的字符串表示。

    参数：
        tool_call: OpenAI 格式的单个工具调用

    返回：
        格式化后的字符串
    """
    function = tool_call.get("function", {})
    function_name = function.get("name", "")

    try:
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            args_dict = json.loads(arguments)
        else:
            args_dict = arguments

        args_str = ", ".join([f"{k}={repr(v)}" for k, v in args_dict.items()])
        return f"{function_name}({args_str})"

    except Exception:
        return f"{function_name}()"


def capture_and_print_score_files(
    score_dir: Path,
    model_name: str,
    test_category: str,
    eval_type: str,
):
    """
    读取并打印写入 score_dir 的评分文件内容。

    参数：
        score_dir: 评分文件所在目录
        model_name: 模型名称
        test_category: 测试类别
        eval_type: 评测类型（relevance/multi_turn/single_turn）
    """
    try:
        print(f"\n=== {eval_type.upper()} Evaluation Result Files ===")
        print(f"Model: {model_name}")
        print(f"Test Category: {test_category}")
        print(f"Evaluation Type: {eval_type}")

        for file_path in score_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(score_dir)
                print(f"\n--- File: {relative_path} ---")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    if (
                        file_path.suffix == ".json"
                        or content.strip().startswith("{")
                        or content.strip().startswith("[")
                    ):
                        try:
                            lines = content.strip().split("\n")
                            formatted_lines = []
                            for line in lines:
                                if line.strip():
                                    parsed = json.loads(line)
                                    formatted_lines.append(
                                        json.dumps(
                                            parsed,
                                            ensure_ascii=False,
                                            indent=2,
                                        ),
                                    )
                            content = "\n".join(formatted_lines)
                        except json.JSONDecodeError:
                            pass

                    print(content)

                except UnicodeDecodeError:
                    print(f"[Binary file, size: {file_path.stat().st_size} bytes]")
                except Exception as e:
                    print(f"[Error reading file: {str(e)}]")

        print(f"=== {eval_type.upper()} Evaluation Result Files End ===\n")

    except Exception as e:
        print(f"Error capturing evaluation result files: {str(e)}")


def extract_tool_schema(tools):
    """重整工具 schema 格式。"""
    for i in range(len(tools)):  # pylint: disable=C0200
        tools[i]["function"].pop("response")
    return tools
