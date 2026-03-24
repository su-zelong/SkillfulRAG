#!/bin/bash
# ==========================================================
# SkillfulRAG: 离线环境一键部署脚本
# ==========================================================

set -e

# --- 全局默认变量 ---
IMAGE_NAME="skillful-rag"
VERSION="latest"
CONTAINER_NAME="rag_engine"

# 默认挂载路径 (当前目录下的对应文件夹)
TAR_PATH="./skillful_rag_offline_${VERSION}.tar"
DATA_PATH="$(pwd)/data"
LOG_PATH="$(pwd)/logs"
CONFIG_PATH="$(pwd)/config.yaml"

# 模型相关变量 (默认为空)
EMBED_MODEL=""
EMBED_API=""
EMBED_KEY=""
LLM_MODEL=""
LLM_API=""
LLM_KEY=""
RERANK_MODEL=""
RERANK_API=""
RERANK_KEY=""

# --- 函数定义 ---

log_info() { echo -e "\n🟢 [INFO] $1"; }
log_err()  { echo -e "\n🔴 [ERROR] $1"; exit 1; }

# 1. 解析命令行参数
parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --tar_path) TAR_PATH="$2"; shift ;;
            --data_path) DATA_PATH="$2"; shift ;;
            --log_path) LOG_PATH="$2"; shift ;;
            --config) CONFIG_PATH="$2"; shift ;;
            
            --embed_model_name) EMBED_MODEL="$2"; shift ;;
            --embed_api) EMBED_API="$2"; shift ;;
            --embed_api_key) EMBED_KEY="$2"; shift ;;
            
            --llm_model_name) LLM_MODEL="$2"; shift ;;
            --llm_api) LLM_API="$2"; shift ;;
            --llm_api_key) LLM_KEY="$2"; shift ;;
            
            --rerank_model_name) RERANK_MODEL="$2"; shift ;;
            --rerank_api) RERANK_API="$2"; shift ;;
            --rerank_api_key) RERANK_KEY="$2"; shift ;;
            
            -h|--help)
                echo "用法: ./deploy_offline.sh [选项]"
                echo "核心参数:"
                echo "  --tar_path <path>       离线镜像包路径"
                echo "  --data_path <dir>       数据持久化目录"
                echo "  --llm_api_key <key>     LLM 模型秘钥"
                exit 0
                ;;
            *) log_err "未知参数: $1" ;;
        esac
        shift
    done
}

# 2. 检查基础环境与文件
check_env() {
    log_info "检查部署环境..."
    command -v docker &> /dev/null || log_err "未检测到 Docker，请先安装。"
    
    if [ ! -f "$TAR_PATH" ]; then
        log_err "未找到镜像包: $TAR_PATH\n请使用 --tar_path 指定正确的路径。"
    fi

    # 自动创建挂载目录，防止 docker run 报错
    mkdir -p "$DATA_PATH" "$LOG_PATH"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        log_info "未找到 $CONFIG_PATH，创建一个空的配置文件..."
        touch "$CONFIG_PATH"
    fi
}

# 3. 载入离线镜像
load_image() {
    log_info "正在载入离线镜像 (这可能需要几分钟)..."
    docker load < "${TAR_PATH}"
}

# 4. 动态生成环境变量文件
generate_env_file() {
    log_info "正在安全生成环境变量文件..."
    ENV_FILE=".env.rag.generated"
    
    # 将用户传入的参数写入临时文件，供 docker 挂载
    cat <<EOF > "$ENV_FILE"
EMBED_MODEL_NAME=${EMBED_MODEL}
EMBED_API_BASE=${EMBED_API}
EMBED_API_KEY=${EMBED_KEY}
LLM_MODEL_NAME=${LLM_MODEL}
OPENAI_API_BASE=${LLM_API}
OPENAI_API_KEY=${LLM_KEY}
RERANK_MODEL_NAME=${RERANK_MODEL}
RERANK_API_BASE=${RERANK_API}
RERANK_API_KEY=${RERANK_KEY}
EOF
}

# 5. 启动容器
run_container() {
    log_info "正在清理旧容器 (如有)..."
    docker rm -f "${CONTAINER_NAME}" &> /dev/null || true

    log_info "🚀 正在启动 SkillfulRAG 容器..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --restart always \
        -v "${DATA_PATH}:/app/data" \
        -v "${LOG_PATH}:/app/logs" \
        -v "${CONFIG_PATH}:/app/config.yaml" \
        --env-file ".env.rag.generated" \
        "${IMAGE_NAME}:${VERSION}"

    log_info "✅ 部署成功！容器已在后台运行。"
    echo "=========================================================="
    echo "📂 数据目录: $DATA_PATH"
    echo "📄 日志目录: $LOG_PATH"
    echo "🔍 查看日志: docker logs -f ${CONTAINER_NAME}"
    echo "=========================================================="
    
    # 阅后即焚，清理包含明文密码的本地环境变量文件
    rm -f ".env.rag.generated"
}

# --- 主函数 ---
main() {
    parse_args "$@"
    check_env
    load_image
    generate_env_file
    run_container
}

# 执行主函数并传入所有命令行参数
main "$@"