from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
import json
import argparse
import sys
import time
import hashlib

# 断点续跑
#python get_results_from_data.py

# 从头重新跑
#python get_results_from_data.py --reset

# 从头跑前10条
#python get_results_from_data.py --reset --limit 10

# 从头跑前100条
#python get_results_from_data.py --reset --limit 100

# 从第200条开始跑100条
#python get_results_from_data.py --reset --start 200 --limit 100

# 每条语料跑 3 次成功再停（取平均/评估用），每个目标最多重试 3 次
# 最坏情况下 = 3 × 3 = 9 次 API 调用
#python get_results_from_data.py --reset --limit 100 --success-attempts 3

# 后台运行
#nohup python get_results_from_data.py --reset --limit 1000 > get_results.log 2>&1 &
# =========================
# 参数
# =========================
parser = argparse.ArgumentParser(
    description="Intent批量测试工具"
)

parser.add_argument(
    "--start",
    type=int,
    default=0,
    help="从第几条待执行数据开始"
)

parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="最多执行多少条"
)

parser.add_argument(
    "--reset",
    action="store_true",
    help="忽略历史结果，从头重新执行"
)

parser.add_argument(
    "--question",
    type=str,
    default=None,
    help="单条快速测试：传入一条语料，调用 API 后只把意图 JSON 打到终端，不写 xlsx",
)

parser.add_argument(
    "--port",
    type=int,
    default=13700,
    help="API 服务端口，默认 13700（与 chatbi_server.sh 默认端口一致）",
)

parser.add_argument(
    "--logs-dir",
    type=str,
    default=None,
    help="日志输出根目录（默认 test_eval/logs）；每条语料生成一个 .log 文件",
)

parser.add_argument(
    "--server-log",
    type=str,
    default=None,
    help="服务端 logger 输出的文件路径（默认 logs/server.<port>.log）。"
         "脚本会在每次 API 请求后按字节位置切片，把该请求期间的日志写入对应问题的 .log 文件。"
         "传 \"\" 或 --no-server-log 关闭此功能。",
)

parser.add_argument(
    "--no-server-log",
    action="store_true",
    help="明确关闭服务端日志捕获（等效于 --server-log \"\"）",
)

parser.add_argument(
    "--server-log-wait",
    type=float,
    default=2.0,
    help="请求返回后等待服务端 logger 落盘的最长时间（秒），默认 2.0。",
)

parser.add_argument(
    "--success-attempts",
    type=int,
    default=1,
    help=(
        "每条语料目标成功次数（intents 非空、非 ERROR 视为成功），默认 1。"
        "每个成功目标允许最多 MAX_ATTEMPTS 次重试，"
        "单条语料最坏尝试数 = --success-attempts × MAX_ATTEMPTS，"
        "最好尝试数 = --success-attempts（每次都一次成功）。"
        "xlsx 输出 try1..tryN 一一对应每个成功目标，"
        "外加一列 重试次数 记录本条语料的总重试数。"
    ),
)

args = parser.parse_args()

START = args.start
LIMIT = args.limit
RESET = args.reset
SINGLE_QUESTION = (args.question or "").strip() or None
PORT = args.port

# =========================
# 文件路径
# =========================
BASE_DIR = Path(__file__).resolve().parent

LOGS_ROOT = Path(args.logs_dir).resolve() if args.logs_dir else (BASE_DIR / "logs")

# 服务端日志路径：默认 logs/server.<port>.log（与 chatbi_server.sh 一致）
if args.no_server_log:
    SERVER_LOG_PATH = None
elif args.server_log is None:
    SERVER_LOG_PATH = (BASE_DIR.parent / "logs" / f"server.{PORT}.log")
else:
    raw = args.server_log
    if raw == "":
        SERVER_LOG_PATH = None
    else:
        SERVER_LOG_PATH = Path(raw).resolve()

SERVER_LOG_WAIT = max(0.0, float(args.server_log_wait))

INPUT_FILE = (
    BASE_DIR
    / "语料标准.xlsx"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / f"intent_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
)

print("INPUT:", INPUT_FILE)
print("OUTPUT:", OUTPUT_FILE)

# =========================
# 重试配置
# =========================
MAX_ATTEMPTS = 3  # 单个成功目标允许的最大重试次数（空列表 / 异常都触发重试）

# 每条语料目标成功次数（命令行可覆盖）
# 两层重试模型：
#   外层 (SUCCESS_ATTEMPTS)：要拿到的成功样本数
#   内层 (MAX_ATTEMPTS)：    每个成功样本最多重试几次
# 单条语料总尝试数 = SUCCESS_ATTEMPTS * MAX_ATTEMPTS
SUCCESS_ATTEMPTS = max(1, int(args.success_attempts))
COLUMN_COUNT = SUCCESS_ATTEMPTS  # xlsx 主体列数 = 成功目标数


# =========================
# Tee 输出流（stdout / stderr 双重目的地：终端 + 当前问题日志文件）
# =========================
class TeeStream:
    """将所有 write 镜像到原始终端流 + 当前激活的日志文件句柄。

    任务级别创建一个全局实例，把 sys.stdout / sys.stderr 换成本类的实例；
    每条语料开始时调用 set_file(path) 切换文件，结束时调用 set_file(None)。
    """

    def __init__(self, original_stream):
        self._original = original_stream
        self._fh = None
        # 进度条 / print 头可能调用 isatty()
        self._isatty = original_stream.isatty()

    def set_file(self, fh):
        # 关闭旧文件（如果有）
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
        self._fh = fh

    def _write_all(self, s):
        if not s:
            return
        try:
            self._original.write(s)
            self._original.flush()
        except Exception:
            pass
        if self._fh is not None:
            try:
                self._fh.write(s)
                self._fh.flush()
            except Exception:
                pass

    def write(self, s):
        self._write_all(s)

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass
        if self._fh is not None:
            try:
                self._fh.flush()
            except Exception:
                pass

    def isatty(self):
        return self._isatty

    # tqdm / requests / pandas 等可能调用以下方法，转发到底层流
    def writelines(self, lines):
        for line in lines:
            self._write_all(line)

    def __getattr__(self, name):
        # 未实现的方法都从原始终端流取
        return getattr(self._original, name)

# =========================
# API配置
# =========================
API_URL = f"http://localhost:{PORT}/v1/query"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": "chatbi_sk_demo"
}

# =========================
# 单条快速测试模式：跳过 xlsx 读写，直接 POST + 打印
# =========================
if SINGLE_QUESTION is not None:
    print("\n====================")
    print("单条快速测试模式")
    print("====================\n")
    print("问题:", SINGLE_QUESTION)
    payload = {
        "question": SINGLE_QUESTION,
        "database_id": "life_insurance",
        "intent_only": True
    }
    try:
        start_time = time.perf_counter()
        resp = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )
        end_time = time.perf_counter()
        print("status =", resp.status_code)
        resp.raise_for_status()
        data = resp.json()
        # 完整响应打终端（indent=2，易读；包含 intents 等全部字段）
        print("--- 完整响应 (indent=2) ---")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        sys.stdout.flush()
        intents = data.get("intents", [])
        print(f"--- intents JSON 字符串 --- {end_time - start_time:.2f}s")
        print(json.dumps(intents, ensure_ascii=False))
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.stdout.flush()
        # 非零退出，方便在脚本里检测失败
        sys.exit(1)
    sys.exit(0)

# =========================
# 会话日志目录（按启动时间戳）
# =========================
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = LOGS_ROOT / f"run_{RUN_TIMESTAMP}"
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
print(f"RUN_LOG_DIR: {RUN_LOG_DIR}")

# 启动 tee：批量模式下所有 print / tqdm 都会同步落到会话级日志
_ORIG_STDOUT = sys.stdout
_ORIG_STDERR = sys.stderr
_STDOUT_TEE = TeeStream(_ORIG_STDOUT)
_STDERR_TEE = TeeStream(_ORIG_STDERR)
sys.stdout = _STDOUT_TEE
sys.stderr = _STDERR_TEE


# =========================
# 服务端日志捕获
# =========================
# 按文件字节位置切片是最简单的实现：每次 API 请求记 pos_before；
# 响应后轮询直到 pos_after 稳定一段时间，读出 [pos_before, pos_after) 作为该请求的日志片段。
# 串行调用场景下位置递增、不会重叠，不需要 trace_id 匹配。
def _safe_filesize(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0
    except Exception:
        return 0


def _wait_for_log_settle(path: Path, prev_size: int, max_wait: float) -> int:
    """轮询直到文件大小连续 2 次未变（以 50ms 为间隔），或超时。返回最终大小。"""
    if max_wait <= 0:
        return _safe_filesize(path)
    deadline = time.monotonic() + max_wait
    stable_prev = prev_size
    while time.monotonic() < deadline:
        time.sleep(0.05)
        cur = _safe_filesize(path)
        if cur == stable_prev and cur > 0:
            # 保持稳定：再确认一次
            time.sleep(0.05)
            if _safe_filesize(path) == cur:
                return cur
        stable_prev = cur
    return _safe_filesize(path)


def read_server_log_chunk(
    log_path: Path,
    pos_before: int,
    max_wait: float = 2.0,
) -> str:
    """读取 [pos_before, end) 区间的服务端日志内容。
    读取失败（文件不存在、权限不足等）返回空串。
    """
    if log_path is None:
        return ""
    try:
        end_pos = _wait_for_log_settle(log_path, pos_before, max_wait)
        if end_pos <= pos_before:
            return ""
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(pos_before)
            return f.read(end_pos - pos_before)
    except Exception as e:
        return f"[read_server_log_chunk failed: {e}]"


def write_server_log_to_qlog(
    q_log_fh,
    attempt: int,
    pos_before: int,
    pos_after: int,
    chunk: str,
):
    """把服务端日志片段写到当前问题的 .log 文件中。
    直接走 fh.write，不走 tee（否则会再被本脚本的 print 镜像到终端）。
    """
    if q_log_fh is None:
        return
    n_bytes = max(0, pos_after - pos_before)
    sep_before = f"\n----- SERVER LOG [try{attempt}] bytes {pos_before}->{pos_after} (+{n_bytes}) -----\n"
    sep_after = f"\n----- /SERVER LOG [try{attempt}] -----\n"
    try:
        q_log_fh.write(sep_before)
        if chunk:
            q_log_fh.write(chunk)
        else:
            # 即便没新内容也写 marker，方便排查"为什么这条 .log 里没服务端日志"
            q_log_fh.write("(no new server log lines in this try)\n")
        q_log_fh.write(sep_after)
        q_log_fh.flush()
    except Exception as e:
        # 不能让日志写失败把脚本弄崩
        try:
            sys.__stderr__.write(f"write_server_log_to_qlog failed: {e}\n")
        except Exception:
            pass


# =========================
# 读取原始数据
# =========================
df = pd.read_excel(INPUT_FILE)

unique_questions = list(
    dict.fromkeys(
        df["语料"]
        .fillna("")
        .astype(str)
    )
)

print(f"原始条数: {len(df)}")
print(f"去重后条数: {len(unique_questions)}")

# =========================
# 历史结果
# =========================
# result_map[q]   = [try1, try2, ...]   每个成功目标一次的代表结果
# duration_map[q] = [d1, d2, ...]       对应那次尝试的耗时
# retry_map[q]    = [r1, r2, ...]       该目标用了多少次（1 = 一次成功）
result_map = {}
duration_map = {}
retry_map = {}
done_questions = set()

if RESET:

    print("\n====================")
    print("RESET模式")
    print("忽略历史结果")
    print("====================\n")

else:

    if OUTPUT_FILE.exists():

        print(f"发现历史结果文件: {OUTPUT_FILE}")

        old_df = pd.read_excel(OUTPUT_FILE)
        is_new_format = "try1" in old_df.columns

        # 新版：try* 列的实际数量（适配历史文件用 3 列、新文件用 N 列）
        old_try_count = sum(
            1 for c in old_df.columns
            if isinstance(c, str) and c.startswith("try") and c[3:].isdigit()
        ) if is_new_format else 0

        # 是否有「重试次数」列（新版才有；旧版假设该条语料 0 次重试）
        has_retry_col = "重试次数" in old_df.columns

        for _, row in old_df.iterrows():

            q = str(row["语料"])

            if is_new_format and old_try_count > 0:
                # 新版：try1..tryK + 耗时1..耗时K（K 取历史文件实际列数）
                result_map[q] = [
                    row.get(f"try{i}", "")
                    for i in range(1, old_try_count + 1)
                ]
                duration_map[q] = [
                    row.get(f"耗时{i}", "")
                    for i in range(1, old_try_count + 1)
                ]
            elif "结果" in old_df.columns:
                # 旧版（无 try* 列）：把单列 结果/耗时 视为 try1/耗时1
                result_map[q] = [row.get("结果", "")]
                duration_map[q] = [row.get("耗时", "")]
            else:
                # 旧版也没有 结果 列：跳过该行
                continue

            # 把所有列 pad 到当前 COLUMN_COUNT（少则补空串，多则截断）
            while len(result_map[q]) < COLUMN_COUNT:
                result_map[q].append("")
            while len(duration_map[q]) < COLUMN_COUNT:
                duration_map[q].append("")
            result_map[q] = result_map[q][:COLUMN_COUNT]
            duration_map[q] = duration_map[q][:COLUMN_COUNT]

            # 「重试次数」是单值（int）：若该列存在则取整数值；否则记 0
            # （旧版 try1..try3 没有这列，按"已成功 0 重试"处理，强制重跑）
            if has_retry_col:
                try:
                    retry_map[q] = int(row.get("重试次数", 0))
                except (ValueError, TypeError):
                    retry_map[q] = 0
            else:
                retry_map[q] = 0

            # 成功次数达到本次目标 → 视为已完成；否则仍需重跑
            succ_n = sum(
                1 for r in result_map[q]
                if pd.notna(r)
                and str(r).strip() != ""
                and not str(r).startswith("ERROR:")
            )
            if succ_n >= SUCCESS_ATTEMPTS:
                done_questions.add(q)

        print(f"已完成: {len(done_questions)}")

# =========================
# 构造待执行列表
# =========================
pending_questions = [
    q
    for q in unique_questions
    if q not in done_questions
]

print(f"待执行总数: {len(pending_questions)}")

# 起始位置
if START > 0:
    pending_questions = pending_questions[START:]

# 限制数量
if LIMIT is not None:
    pending_questions = pending_questions[:LIMIT]

print("\n====================")
print("运行参数")
print("reset =", RESET)
print("start =", START)
print("limit =", LIMIT)
print("max_attempts (单目标重试上限) =", MAX_ATTEMPTS)
print("success_attempts (目标成功次数) =", SUCCESS_ATTEMPTS)
print(f"最坏尝试数 = success_attempts × max_attempts = {SUCCESS_ATTEMPTS * MAX_ATTEMPTS}")
print("本次执行 =", len(pending_questions))
print("====================\n")

# =========================
# 请求接口
# =========================
for idx, question in enumerate(
    tqdm(pending_questions),
    start=1
):

    # =========================
    # 打开本条语料对应的日志文件
    # 命名规则：<行号>__<qhash[:8]>.log
    # qhash 用 md5 前 8 位，避免文件名包含特殊字符 / 过长
    # =========================
    line_no = START + idx
    q_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    q_log_path = RUN_LOG_DIR / f"{line_no:04d}__{q_hash}.log"
    q_log_fh = open(q_log_path, "w", encoding="utf-8")
    _STDOUT_TEE.set_file(q_log_fh)
    _STDERR_TEE.set_file(q_log_fh)

    # 文件头：方便后面 grep / 索引
    print(
        f"\n[{idx}/{len(pending_questions)}]"
    )

    print("问题:", question)
    print(f"LOG_FILE: {q_log_path}")
    if SERVER_LOG_PATH is not None:
        print(f"SERVER_LOG: {SERVER_LOG_PATH}")
        # 把基线位置记下来：本条问题开始前的服务端日志大小
        # 多个 try 的窗口都是在它之后的累计增量
        server_log_baseline = _safe_filesize(SERVER_LOG_PATH)
    else:
        server_log_baseline = None

    payload = {
        "question": question,
        "database_id": "life_insurance",
        "intent_only": True
    }

    target_results = []    # 长度 = SUCCESS_ATTEMPTS；每个目标一次"代表性"结果
    target_durations = []  # 同上，每次代表目标最后一次尝试的耗时
    target_retries = []    # 每个目标实际试了几次（1 = 一次成功）
    success_count = 0      # 本条语料累计成功目标数
    total_attempts = 0     # 本条语料累计实际 API 调用次数

    # 两层循环：外层是要拿到的成功样本数，内层是每个样本允许的最大重试数
    # 单条语料最多 SUCCESS_ATTEMPTS * MAX_ATTEMPTS 次 API 调用
    for s_idx in range(1, SUCCESS_ATTEMPTS + 1):

        this_target_succeeded = False
        this_target_last_result = ""
        this_target_last_duration = 0.0

        for attempt in range(1, MAX_ATTEMPTS + 1):

            total_attempts += 1

            start_time = time.perf_counter()
            result = None
            this_success = False

            # 记录本 try 开始前的服务端日志位置（字节偏移）
            if SERVER_LOG_PATH is not None:
                server_log_pos_before = _safe_filesize(SERVER_LOG_PATH)
            else:
                server_log_pos_before = None

            try:

                resp = requests.post(
                    API_URL,
                    json=payload,
                    headers=headers,
                    timeout=60
                )

                print(
                    f"  [目标{s_idx}/{SUCCESS_ATTEMPTS} 第{attempt}/{MAX_ATTEMPTS}次] "
                    f"status = {resp.status_code}"
                )

                resp.raise_for_status()

                data = resp.json()

                intents = data.get(
                    "intents",
                    []
                )

                result = json.dumps(
                    intents,
                    ensure_ascii=False
                )

                if intents:
                    print(
                        f"  [目标{s_idx}/{SUCCESS_ATTEMPTS} 第{attempt}/{MAX_ATTEMPTS}次] "
                        f"成功 (intents={len(intents)})"
                    )
                    this_success = True
                else:
                    print(
                        f"  [目标{s_idx}/{SUCCESS_ATTEMPTS} 第{attempt}/{MAX_ATTEMPTS}次] "
                        f"intents 为空，将重试"
                    )

            except Exception as e:

                result = f"ERROR: {e}"

                print(
                    f"  [目标{s_idx}/{SUCCESS_ATTEMPTS} 第{attempt}/{MAX_ATTEMPTS}次] "
                    f"{result}"
                )

            end_time = time.perf_counter()
            this_target_last_result = result
            this_target_last_duration = round(end_time - start_time, 3)

            # =========================
            # 把本 try 期间服务端写入的日志片段附加到本问题的 .log 文件
            # =========================
            if server_log_pos_before is not None and SERVER_LOG_PATH is not None:
                server_log_pos_after = _safe_filesize(SERVER_LOG_PATH)
                if server_log_pos_after > server_log_pos_before:
                    chunk = read_server_log_chunk(
                        SERVER_LOG_PATH,
                        server_log_pos_before,
                        SERVER_LOG_WAIT,
                    )
                    # 写文件内部仍用全局尝试编号，方便逐次定位
                    write_server_log_to_qlog(
                        q_log_fh,
                        total_attempts,
                        server_log_pos_before,
                        server_log_pos_after,
                        chunk,
                    )
                    print(
                        f"  [目标{s_idx}/{SUCCESS_ATTEMPTS} 第{attempt}/{MAX_ATTEMPTS}次] "
                        f"已捕获服务端日志 {server_log_pos_before}->{server_log_pos_after} "
                        f"(+{server_log_pos_after - server_log_pos_before} bytes)"
                    )

            if this_success:
                this_target_succeeded = True
                success_count += 1
                break
            # 失败则继续内层下一次重试

        target_results.append(this_target_last_result)
        target_durations.append(this_target_last_duration)
        target_retries.append(attempt)  # 本目标最终用了几次（无论成败）

        # 本目标用尽 MAX_ATTEMPTS 次仍失败 → 继续试下一个目标（不提前终止整条语料）
        if not this_target_succeeded:
            print(
                f"  [目标{s_idx}/{SUCCESS_ATTEMPTS}] 已用尽 {MAX_ATTEMPTS} 次，"
                f"继续下一个目标"
            )

    result_map[question] = target_results
    duration_map[question] = target_durations
    retry_map[question] = total_attempts  # 单值：本条语料总 API 调用次数

    print(
        f"  [完成] 成功 {success_count}/{SUCCESS_ATTEMPTS} 个目标，"
        f"实际执行 {total_attempts}/{SUCCESS_ATTEMPTS * MAX_ATTEMPTS} 次 API"
        f"，每目标 {' '.join(str(r) for r in target_retries)} 次"
    )

    # =========================
    # 关闭本条语料的日志文件
    # =========================
    _STDOUT_TEE.set_file(None)
    _STDERR_TEE.set_file(None)

# =========================
# 回填结果
# =========================
q_list = (
    df["语料"]
    .fillna("")
    .astype(str)
    .tolist()
)

for i in range(1, COLUMN_COUNT + 1):

    df[f"try{i}"] = [
        result_map.get(q, [""] * COLUMN_COUNT)[i - 1]
        if q in result_map
        else ""
        for q in q_list
    ]

    df[f"耗时{i}"] = [
        duration_map.get(q, [""] * COLUMN_COUNT)[i - 1]
        if q in duration_map
        else ""
        for q in q_list
    ]

# 「重试次数」单值列：本条语料本次累计 API 调用次数
df["重试次数"] = [
    retry_map.get(q, 0)
    for q in q_list
]

# =========================
# 输出Excel
# =========================
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_excel(
    OUTPUT_FILE,
    index=False
)

# =========================
# 会话索引文件：记录本次任务参数 + 落点 Excel 路径，
# 方便通过日志文件反查对应的 Excel 行。
# =========================
INDEX_FILE = RUN_LOG_DIR / "README.txt"
with open(INDEX_FILE, "w", encoding="utf-8") as idx_fh:
    idx_fh.write(f"run_timestamp : {RUN_TIMESTAMP}\n")
    idx_fh.write(f"reset         : {RESET}\n")
    idx_fh.write(f"start         : {START}\n")
    idx_fh.write(f"limit         : {LIMIT}\n")
    idx_fh.write(f"port          : {PORT}\n")
    idx_fh.write(f"max_attempts  : {MAX_ATTEMPTS}  # 单目标最大重试数\n")
    idx_fh.write(f"success_attempts: {SUCCESS_ATTEMPTS}  # 每条语料目标成功次数\n")
    idx_fh.write(f"worst_attempts: {SUCCESS_ATTEMPTS * MAX_ATTEMPTS}\n")
    idx_fh.write(f"output_xlsx   : {OUTPUT_FILE}\n")
    idx_fh.write(f"total_records : {len(df)}\n")
    idx_fh.write(f"executed_now  : {len(pending_questions)}\n")
    idx_fh.write("log_naming    : <行号>__<qhash前8位>.log\n")

# =========================
# 完成
# =========================
print("\n====================")
print("处理完成")
print("总记录数:", len(df))
print("本次执行数:", len(pending_questions))
print("输出文件:", OUTPUT_FILE)
print("日志目录:", RUN_LOG_DIR)
print("====================")

# 恢复原始终端流
sys.stdout = _ORIG_STDOUT
sys.stderr = _ORIG_STDERR