#!/usr/bin/env bash
# =============================================================================
# ChatBI 服务器管理脚本
# =============================================================================
# 用法：
#   ./chatbi_server.sh start [PORT]      # 启动；PORT 默认 13700；若被占则自增到下一个空闲端口（最多尝试 13700-13709）
#   ./chatbi_server.sh restart [PORT]     # 重启；PORT 默认 13700；先 lsof 杀进程再启动（端口保持不变）
#   ./chatbi_server.sh stop [PORT]        # 停止
#   ./chatbi_server.sh status [PORT]      # 查看运行状态
#
# 环境变量（可选）：
#   CHATBI_PYTHON     指向要使用的 python 解释器；不设置时自动选择 conda 环境的 python
#   CHATBI_CONDA_ENV  目标 conda 环境名（默认 chatbi）
#   CHATBI_HOST       监听地址（默认 0.0.0.0）
#
# 运行时产物：
#   logs/server.<port>.log    uvicorn stdout/stderr（仅 stop/restart 失败排查时手动查看）
#   logs/server.<port>.pid    兼容旧版遗留文件（前台模式不再写入）
# =============================================================================

set -u

# 强制 C/POSIX locale，防止 bash 把中文/全角字符当成变量名的一部分
# （如 $port 后紧跟 `（` U+FF08 时，bash 误把 `port\xef` 当成一个变量）
export LC_ALL=C
export LANG=C

# ---------------- 工具函数（先定义，后面的初始化和 resolve_python 要用到） ----------------
log()  { echo "[$(date '+%F %T')] $*"; }
err()  { echo "[$(date '+%F %T')] ERROR: $*" >&2; }

# ---------------- 配置 ----------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

DEFAULT_PORT=13700
PORT_START=13700
PORT_END=13709
HOST="${CHATBI_HOST:-0.0.0.0}"

# 默认 conda 环境名（可通过环境变量 CHATBI_CONDA_ENV 覆盖）
# 脚本会按这个名称在 conda 里查找 python 解释器；改这里就能切换环境
DEFAULT_CONDA_ENV="chatbi"
CONDA_ENV="${CHATBI_CONDA_ENV:-$DEFAULT_CONDA_ENV}"

# ---------------- Python 解释器自动解析 ----------------
# 优先级：
#   1) 显式环境变量 CHATBI_PYTHON（最高优先级）
#   2) 当前 PATH 上的 python，如果它本身就在 CONDA_ENV 指定的 conda env 里
#   3) 通过 `conda run -n "$CONDA_ENV" which python` 解析（依赖 conda 已初始化）
#   4) 常见 conda 安装路径下的 $base/envs/$CONDA_ENV/bin/python
#   5) 兜底：PATH 上的 python
resolve_python() {
    # 1) 显式指定
    if [ -n "${CHATBI_PYTHON:-}" ]; then
        if [ -x "$CHATBI_PYTHON" ] || command -v "$CHATBI_PYTHON" >/dev/null 2>&1; then
            echo "$CHATBI_PYTHON"
            return 0
        fi
        err "CHATBI_PYTHON='$CHATBI_PYTHON' 不可执行，将尝试自动检测"
    fi

    local py="" candidate="" env_name="$CONDA_ENV"

    # 2) 当前 python 是否已在目标 conda env 内
    #    通过 sys.prefix 的最后一段路径名（conda 把 env 装在 $base/envs/<name>）判断，
    #    比 'chatbi' in sys.prefix 之类的子串匹配更准确（避免误匹配到路径里恰好含
    #    env 名 的目录）
    if command -v python >/dev/null 2>&1; then
        local cur_prefix
        cur_prefix="$(python -c 'import sys,os; print(os.path.basename(sys.prefix))' 2>/dev/null)"
        if [ -n "$cur_prefix" ] && [ "$cur_prefix" = "$env_name" ]; then
            command -v python
            return 0
        fi
    fi

    # 3) 用 conda 解析
    if command -v conda >/dev/null 2>&1; then
        candidate="$(conda run -n "$env_name" which python 2>/dev/null | tail -n 1 | tr -d '[:space:]')"
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
        # 回退：直接从 conda env list 解析 # 注释行下的路径
        candidate="$(conda env list 2>/dev/null \
            | sed -n "s/^${env_name}[[:space:]]\{1,\}\*\{0,1\}[[:space:]]\{1,\}\([^[:space:]]*\).*/\1/p" \
            | head -n 1)"
        if [ -n "$candidate" ] && [ -x "$candidate/bin/python" ]; then
            echo "$candidate/bin/python"
            return 0
        fi
    fi

    # 4) 常见 conda 安装路径
    for base in /opt/miniconda3 "$HOME/miniconda3" "$HOME/opt/miniconda3" \
                /opt/anaconda3 "$HOME/anaconda3" \
                /opt/conda "$HOME/.conda" \
                /usr/local/miniconda3 "$HOME/.local/miniconda3"; do
        if [ -x "$base/envs/$env_name/bin/python" ]; then
            echo "$base/envs/$env_name/bin/python"
            return 0
        fi
    done

    # 5) 兜底
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    err "找不到任何可用的 python 解释器"
    return 1
}

PYTHON_BIN="$(resolve_python)" || exit 1
if [ -n "${CHATBI_PYTHON:-}" ]; then
    log "使用 CHATBI_PYTHON: $PYTHON_BIN"
else
    # 打印当前 python 所在 conda env，方便确认
    _env_name="$("$PYTHON_BIN" -c "import sys,os; p=sys.prefix; print(os.path.basename(p) if p else '(none)')" 2>/dev/null || echo '?')"
    log "自动选择 python: $PYTHON_BIN (env=$_env_name)"
    unset _env_name
fi

DATA_DIR="$PROJECT_ROOT/logs"
mkdir -p "$DATA_DIR"

pid_file()    { echo "$DATA_DIR/server.$1.pid"; }
log_file()    { echo "$DATA_DIR/server.$1.log"; }

# 通过 lsof 查占用指定端口的 PID（兼容没有 lsof 的环境，回退到 ss）
pids_on_port() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:"$port" 2>/dev/null
    elif command -v ss >/dev/null 2>&1; then
        ss -tlnp 2>/dev/null | awk -v p=":$port" '$4 ~ p {print $0}' \
            | grep -oP 'pid=\K[0-9]+' | sort -u
    elif command -v fuser >/dev/null 2>&1; then
        fuser "${port}/tcp" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true
    else
        err "需要 lsof / ss / fuser 之一来检测端口占用"
        return 1
    fi
}

is_port_free() {
    local port="$1"
    local pids
    pids="$(pids_on_port "$port" 2>/dev/null | tr -d ' ')"
    [ -z "$pids" ]
}

find_free_port() {
    local start="$PORT_START"
    local end="$PORT_END"
    for ((p = start; p <= end; p++)); do
        if is_port_free "$p"; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

is_running() {
    local port="$1"
    local pf
    pf="$(pid_file "$port")"
    # 优先看 PID 文件（更准确，因为能区分"我们启动的进程" vs "其它占用进程"）
    if [ -f "$pf" ]; then
        local pid
        pid="$(cat "$pf" 2>/dev/null || true)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    # PID 文件丢失但端口还在被监听 ⇒ 也算在跑（可能是手工/别的工具启动的）
    if ! is_port_free "$port"; then
        return 0
    fi
    return 1
}

# 强杀指定端口上的所有进程；优先用 lsof -ti 一次拿齐
kill_port() {
    local port="$1"
    local pids
    pids="$(pids_on_port "$port" 2>/dev/null | tr -d ' \n')"
    if [ -z "$pids" ]; then
        log "端口 $port 当前无进程占用"
        return 0
    fi
    log "发现占用端口 $port 的 PID: $pids"
    # 先优雅
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done
    # 最多等 2 秒
    for _ in 1 2 3 4; do
        sleep 0.5
        pids="$(pids_on_port "$port" 2>/dev/null | tr -d ' \n')"
        [ -z "$pids" ] && break
    done
    # 强杀
    pids="$(pids_on_port "$port" 2>/dev/null | tr -d ' \n')"
    if [ -n "$pids" ]; then
        log "强 kill -9: $pids"
        for pid in $pids; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 0.3
    fi
    # 同时清掉遗留 PID 文件
    rm -f "$(pid_file "$port")"
    log "端口 $port 已释放"
}

# ---------------- 子命令 ----------------
cmd_start() {
    local requested="${1:-$DEFAULT_PORT}"
    local port="$requested"

    if is_running "$port"; then
        err "端口 $port 已经在运行（PID=$(cat "$(pid_file "$port")" 2>/dev/null || pids_on_port "$port" | tr -s ' \n' ' ')）。如需重启请用 restart。"
        return 1
    fi

    # start 模式：只在端口被占时自动找下一个；不主动杀进程
    if ! is_port_free "$port"; then
        local alt
        if alt="$(find_free_port)"; then
            err "端口 $port 被占，自动改用 $alt（可用范围 ${PORT_START}-${PORT_END}）"
            port="$alt"
        else
            err "端口 $port 被占，且 ${PORT_START}-${PORT_END} 全部被占；请先 stop 或 restart"
            return 1
        fi
    fi

    local logf
    logf="$(log_file "$port")"
    # 启动时清空旧日志（不保留上次运行的内容）
    : > "$logf"

    log "启动 ChatBI 服务（前台运行）：$HOST:$port（python=$PYTHON_BIN）"
    log "日志同时写入：$logf"
    log "提示：Ctrl-C 或关闭终端窗口即可停止服务"

    # 前台运行 + 终端输出与日志文件双写
    # 方式：把 uvicorn 的 stdout/stderr 都 pipe 给 `tee -a`，tee 一份回终端、一份到日志
    # - uvicorn 与 tee 在同一前台进程组，Ctrl-C 由内核同时发给 shell/uvicorn/tee，
    #   uvicorn 收到 SIGINT 会优雅退出（它自带 handler），tee 收到 SIGPIPE 也退出
    # - 启动前/后的 banner（`log` 调用的）走当前 shell 的 fd，**不会**进日志；
    #   若想 banner 也进日志，得 `exec > >(tee ...)` 改 shell 自身的 fd，但那会把
    #   Ctrl-C 时的清屏/光标控制序列也写进日志，污染严重。uvicorn 自己的输出
    #   （INFO/WARNING/ERROR/异常栈）已足够排查问题。
    # - 不需要 nohup / & / disown，关闭终端 = SIGHUP 给整个进程组，uvicorn 退出
    # - 不用 `exec` 改 shell 自身的 fd，否则后续 trap/log 也会被 tee 反复写，日志翻倍
    trap 'echo; log "正在停止 ChatBI ..."' INT
    # stdbuf -oL -eL 让 tee 立即刷盘（line-buffered），避免 Ctrl-C 后日志丢最后几行
    # macOS 上 stdbuf 不一定可用；tee 在收到 SIGPIPE 时会自然 flush 关闭，足够
    if command -v stdbuf >/dev/null 2>&1; then
        "$PYTHON_BIN" -m uvicorn main:app \
            --host "$HOST" --port "$port" \
            --no-access-log 2>&1 \
            | stdbuf -oL -eL tee -a "$logf"
    else
        "$PYTHON_BIN" -m uvicorn main:app \
            --host "$HOST" --port "$port" \
            --no-access-log 2>&1 \
            | tee -a "$logf"
    fi
    local rc=$?
    trap - INT
    # 退出后追加一条 banner，方便阅读日志时分清多次启动
    {
        echo
        echo "[$(date '+%F %T')] ChatBI 服务已退出（rc=$rc）"
    } >> "$logf"
    return $rc
}

cmd_restart() {
    local port="${1:-$DEFAULT_PORT}"
    log "===== 重启端口 $port ====="
    kill_port "$port"
    # 保险：再确认一次
    sleep 0.3
    cmd_start "$port"
}

cmd_stop() {
    local port="${1:-$DEFAULT_PORT}"
    log "===== 停止端口 $port ====="
    kill_port "$port"
    log "已停止端口 $port"
}

cmd_status() {
    local port="${1:-$DEFAULT_PORT}"
    local pf
    pf="$(pid_file "$port")"
    # 我们启动的进程
    if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
        log "端口 $port 正在运行：PID=$(cat "$pf")  日志=$(log_file "$port")"
        return 0
    fi
    # PID 文件没了但端口仍被监听 ⇒ 别的工具启动的进程
    if ! is_port_free "$port"; then
        local ext
        ext="$(pids_on_port "$port" | tr -s ' \n' ' ' | xargs)"
        log "端口 $port 被外部进程占用：PID=$ext（非本脚本启动）"
        return 0
    fi
    log "端口 $port 未运行"
    if [ -f "$pf" ]; then
        err "残留 PID 文件 $pf（进程已死）"
    fi
    return 1
}

# ---------------- 入口 ----------------
usage() {
    sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
    start)    shift; cmd_start    "${1:-$DEFAULT_PORT}" ;;
    restart)  shift; cmd_restart  "${1:-$DEFAULT_PORT}" ;;
    stop)     shift; cmd_stop     "${1:-$DEFAULT_PORT}" ;;
    status)   shift; cmd_status   "${1:-$DEFAULT_PORT}" ;;
    -h|--help|help|"")
        usage
        ;;
    *)
        err "未知子命令: $1"
        usage
        exit 1
        ;;
esac
