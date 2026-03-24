#!/bin/bash
# ==========================================================
# SkillfulRAG: 离线镜像构建与导出脚本
# 执行路径要求: 请在 scripts/ 目录下执行此脚本
# ==========================================================

set -e # 遇到错误立即退出

# --- 全局变量配置 ---
IMAGE_NAME="skillful-rag"
VERSION="latest"
TAR_NAME="skillful_rag_offline_${VERSION}.tar"
DOCKERFILE_PATH="../Dockerfile"
BUILD_CONTEXT=".." # 构建上下文指向项目根目录

# --- 函数定义 ---

# 1. 打印日志格式化
log_info() { echo -e "\n🟢 [INFO] $1"; }
log_err()  { echo -e "\n🔴 [ERROR] $1"; exit 1; }

# 2. 环境检查
check_env() {
    log_info "正在检查打包环境..."
    command -v docker &> /dev/null || log_err "未检测到 Docker，请先安装。"
    docker info &> /dev/null || log_err "Docker 未运行，请启动 Docker 守护进程。"
    
    if [ ! -f "$DOCKERFILE_PATH" ]; then
        log_err "未找到 Dockerfile: $DOCKERFILE_PATH\n请确保你在 scripts/ 目录下执行此脚本！"
    fi
}

# 3. 构建镜像
build_image() {
    log_info "开始构建镜像: ${IMAGE_NAME}:${VERSION}..."
    # -f 指定 Dockerfile 路径，最后一个参数是构建上下文
    # Docker 会自动在构建上下文中寻找 .dockerignore (即 ../.dockerignore)
    docker build \
        -f "${DOCKERFILE_PATH}" \
        -t "${IMAGE_NAME}:${VERSION}" \
        "${BUILD_CONTEXT}"
    log_info "镜像构建完成！"
}

# 4. 导出镜像包
export_tarball() {
    log_info "正在导出离线镜像包至当前目录: ${TAR_NAME}..."
    docker save "${IMAGE_NAME}:${VERSION}" > "${TAR_NAME}"
    log_info "✅ 导出成功！请将 ${TAR_NAME} 和部署脚本发送至离线环境。"
}

# --- 主函数 ---
main() {
    check_env
    build_image
    export_tarball
}

# 执行主函数
main