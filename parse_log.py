import re
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple

def parse_log_line(line: str) -> Optional[Dict]:
    """解析单行日志，返回结构化信息"""
    # 提取时间戳
    time_match = re.match(r'(\d{2}:\d{2}:\d{2})', line)
    if not time_match:
        return None
    timestamp = time_match.group(1)
    
    # pipeline.orchestrator 提取 id
    orch_match = re.search(r'pipeline\.orchestrator: \[([a-f0-9]+)\]', line)
    if orch_match:
        return {
            'type': 'orchestrator',
            'timestamp': timestamp,
            'id': orch_match.group(1)
        }
    
    # [perf] .*.run 若干s
    perf_match = re.search(r'\[perf\] (\S+\.run) ([\d.]+)s', line)
    if perf_match:
        return {
            'type': 'perf',
            'timestamp': timestamp,
            'component': perf_match.group(1),
            'duration': float(perf_match.group(2))
        }
    
    # llm.router: LLM call started/messages/finished
    llm_start_match = re.search(r'llm\.router: LLM call started', line)
    if llm_start_match:
        return {
            'type': 'llm_start_ts',
            'timestamp': timestamp
        }

    llm_messages_match = re.search(r'llm\.router: LLM call messages', line)
    if llm_messages_match:
        # 确定类型
        intent_type = None
        if '关键词提取' in line:
            intent_type = '关键词提取'
        elif '指标定位' in line:
            intent_type = '指标定位'
        elif '实体意图识别' in line:
            intent_type = '实体意图识别'
        elif '意图检查纠正' in line:
            intent_type = '意图检查纠正'

        if intent_type:
            return {
                'type': 'llm_start',
                'timestamp': timestamp,
                'intent_type': intent_type
            }
        else:
            # 其他 messages 类型，忽略
            return None

    llm_finish_match = re.search(r'llm\.router: LLM call finished', line)
    if llm_finish_match:
        # finished 行不携带 duration，需与最近一次 start 配对计算
        return {
            'type': 'llm_finish',
            'timestamp': timestamp
        }
    
    # embeddings HTTP/1.1 200 OK
    if 'embeddings' in line and 'HTTP/1.1 200 OK' in line:
        return {
            'type': 'embedding',
            'timestamp': timestamp
        }
    
    return None


def _ts_to_seconds(ts: str) -> int:
    """将 HH:MM:SS 时间戳转换为当天秒数"""
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def generate_compact_view(log_lines: List[str]) -> str:
    """生成简洁日志视图"""
    output_lines = []

    i = 0
    n = len(log_lines)
    embedding_count = 0
    embedding_start_time = None
    pending_llm_type = None      # 暂存的 llm start 类型，等待 finish
    pending_llm_start_ts = None  # llm call started 的时间戳（用于算耗时）

    while i < n:
        line = log_lines[i].strip()
        if not line:
            i += 1
            continue

        parsed = parse_log_line(line)

        if not parsed:
            i += 1
            continue

        if parsed['type'] == 'orchestrator':
            output_lines.append(f"{parsed['timestamp']} pipeline.orchestrator: [{parsed['id']}]")
            i += 1

        elif parsed['type'] == 'perf':
            output_lines.append(f"{parsed['timestamp']} [{parsed['component']} {parsed['duration']:.3f}s]")
            i += 1

        elif parsed['type'] == 'embedding':
            if embedding_start_time is None:
                embedding_start_time = parsed['timestamp']
            embedding_count += 1
            i += 1
            # 检查下一行是否也是 embedding
            continue

        elif parsed['type'] == 'llm_start_ts':
            # 记录 started 时间戳，消息体行在下一条
            pending_llm_start_ts = parsed['timestamp']
            i += 1

        elif parsed['type'] == 'llm_start':
            # 收到 messages 行（带意图类型）—— 输出 messages 行
            pending_llm_type = parsed['intent_type']
            output_lines.append(f"{parsed['timestamp']} llm.router: LLM call messages\t{pending_llm_type}")
            i += 1

        elif parsed['type'] == 'llm_finish':
            # 与最近一次 start 配对计算耗时
            if pending_llm_type and pending_llm_start_ts:
                duration = _ts_to_seconds(parsed['timestamp']) - _ts_to_seconds(pending_llm_start_ts)
                if duration < 0:
                    duration = 0
                output_lines.append(f"{parsed['timestamp']} llm.router: LLM call finished\t{duration}s")
            pending_llm_type = None
            pending_llm_start_ts = None
            i += 1

        # 在行索引移动前，检查是否需要输出 embeddings 汇总
        # 如果当前不是 embedding 且之前有累积的 embedding
        if (parsed['type'] != 'embedding' and embedding_count > 0) or (i >= n and embedding_count > 0):
            if embedding_start_time:
                output_lines.append(f"{embedding_start_time} embeddings * {embedding_count}")
            embedding_count = 0
            embedding_start_time = None

    return '\n'.join(output_lines)


def _read_log_file(path: str) -> List[str]:
    """读取单个日志文件，返回行列表"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def _collect_log_files(folder: str) -> List[str]:
    """收集文件夹下所有 .log 文件（按文件名排序）"""
    import os
    entries = sorted(os.listdir(folder))
    log_files = [
        os.path.join(folder, name)
        for name in entries
        if name.endswith('.log') and os.path.isfile(os.path.join(folder, name))
    ]
    return log_files


def _print_usage() -> None:
    """打印使用说明"""
    print(
        "用法:\n"
        "  python parse_log.py                                  # 从 stdin 读取\n"
        "  python parse_log.py <log_file>                       # 解析单个文件，打印到 stdout\n"
        "  python parse_log.py <log_file> -o <output_file>      # 解析单个文件，导出到文件\n"
        "  python parse_log.py <log_folder>                     # 解析文件夹下所有 .log，\n"
        "                                                        #   不同文件间用空行分隔，打印到 stdout\n"
        "  python parse_log.py <log_folder> -o <output_file>    # 解析文件夹并导出到文件",
        file=sys.stderr,
    )


def main():
    """主函数：根据参数选择输入源和输出目标"""
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description="将 pipeline 原始日志解析为简洁视图（LLM call 配对计算耗时，embedding 汇总）。",
        add_help=True,
    )
    parser.add_argument(
        'input',
        nargs='?',
        help='日志文件路径、包含 .log 文件的文件夹路径；不传则从 stdin 读取',
    )
    parser.add_argument(
        '-o', '--output',
        help='导出到指定文件；不传则打印到 stdout',
    )
    args = parser.parse_args()

    sections: List[str] = []

    if args.input is None:
        # stdin 模式
        log_lines = sys.stdin.readlines()
        if not log_lines:
            _print_usage()
            sys.exit(1)
        sections.append(generate_compact_view(log_lines))
    elif os.path.isdir(args.input):
        # 文件夹模式
        log_files = _collect_log_files(args.input)
        if not log_files:
            print(f"在文件夹 {args.input!r} 中未找到任何 .log 文件", file=sys.stderr)
            sys.exit(1)
        for log_file in log_files:
            try:
                log_lines = _read_log_file(log_file)
            except OSError as e:
                print(f"读取 {log_file} 失败: {e}", file=sys.stderr)
                continue
            sections.append(generate_compact_view(log_lines))
    elif os.path.isfile(args.input):
        # 单文件模式
        log_lines = _read_log_file(args.input)
        sections.append(generate_compact_view(log_lines))
    else:
        _print_usage()
        sys.exit(1)

    # 文件夹多文件场景：用空行分隔；输出末尾追加一个换行
    output = '\n\n'.join(sections)
    if output:
        output += '\n'

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
    else:
        sys.stdout.write(output)


if __name__ == '__main__':
    main()