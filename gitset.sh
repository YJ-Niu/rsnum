#!/bin/bash
# ============================================================
# 脚本名称: git-setup-push.sh
# 功能: 管理项目的多个远程仓库，支持自由切换和推送
#       确保 IDE Git 正确识别仓库
# 用法: 
#   ./gitset.sh add <远程名称> <仓库URL>     - 添加远程仓库
#   ./gitset.sh list                         - 列出所有远程仓库
#   ./gitset.sh use <远程名称> [分支名]      - 切换到指定远程并推送
#   ./gitset.sh push <远程名称> [分支名]     - 推送到指定远程
#   ./gitset.sh pull <远程名称> [分支名]     - 从指定远程拉取
#   ./gitset.sh remove <远程名称>            - 删除远程仓库
#   ./gitset.sh --help                      - 显示帮助
# ============================================================

set -e  # 遇到错误立即退出

# ---------- 默认配置 ----------
DEFAULT_BRANCH="master"
# -----------------------------

usage() {
    echo "用法: $0 <命令> [参数...]"
    echo ""
    echo "命令:"
    echo "  add    <远程名称> <仓库URL>   添加新的远程仓库"
    echo "  list                          列出所有已配置的远程仓库"
    echo "  use    <远程名称> [分支名]     切换到指定远程并推送（设置默认）"
    echo "  push   <远程名称> [分支名]     推送到指定远程仓库"
    echo "  pull   <远程名称> [分支名]     从指定远程仓库拉取"
    echo "  remove <远程名称>              删除指定远程仓库"
    echo "  --help                         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 add gitee https://gitee.com/love_develop/rsnum.git"
    echo "  $0 add github https://github.com/user/rsnum.git"
    echo "  $0 list"
    echo "  $0 use gitee master"
    echo "  $0 push github main"
    echo "  $0 pull gitee"
    echo ""
    exit 0
}

# 检查 git
if ! command -v git &> /dev/null; then
    echo "错误: 未找到 git 命令，请先安装 Git"
    exit 1
fi

# 检查是否在 git 仓库中
if [ ! -d ".git" ]; then
    echo "错误: 当前目录不是 Git 仓库"
    echo "请先在项目目录中运行: git init"
    exit 1
fi

# ---------- 命令处理 ----------

case "$1" in
    --help)
        usage
        ;;
        
    add)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "用法: $0 add <远程名称> <仓库URL>"
            echo "示例: $0 add gitee https://gitee.com/love_develop/rsnum.git"
            exit 1
        fi
        REMOTE_NAME="$2"
        REPO_URL="$3"
        
        if git remote get-url "$REMOTE_NAME" &> /dev/null; then
            echo "远程仓库 $REMOTE_NAME 已存在，更新 URL..."
            OLD_URL=$(git remote get-url "$REMOTE_NAME")
            echo "  原地址: $OLD_URL"
            echo "  新地址: $REPO_URL"
            git remote set-url "$REMOTE_NAME" "$REPO_URL"
        else
            echo "添加远程仓库 $REMOTE_NAME..."
            git remote add "$REMOTE_NAME" "$REPO_URL"
        fi
        echo "✓ 远程仓库 $REMOTE_NAME 配置完成: $REPO_URL"
        ;;
        
    list)
        echo "========== 已配置的远程仓库 =========="
        git remote -v
        echo ""
        echo "当前分支: $(git branch --show-current)"
        CURRENT_UPSTREAM=$(git rev-parse --abbrev-ref HEAD@{upstream} 2>/dev/null || echo "未设置")
        echo "当前上游: $CURRENT_UPSTREAM"
        ;;
        
    use)
        if [ -z "$2" ]; then
            echo "用法: $0 use <远程名称> [分支名]"
            echo "示例: $0 use gitee master"
            exit 1
        fi
        REMOTE_NAME="$2"
        BRANCH="${3:-$DEFAULT_BRANCH}"
        
        # 检查远程是否存在
        if ! git remote get-url "$REMOTE_NAME" &> /dev/null; then
            echo "错误: 远程仓库 $REMOTE_NAME 不存在"
            echo "请先添加: $0 add $REMOTE_NAME <仓库URL>"
            exit 1
        fi
        
        REPO_URL=$(git remote get-url "$REMOTE_NAME")
        echo "=========================================="
        echo "切换到远程仓库: $REMOTE_NAME"
        echo "仓库地址: $REPO_URL"
        echo "目标分支: $BRANCH"
        echo "=========================================="
        
        # 确保本地分支存在
        CURRENT_BRANCH=$(git branch --show-current)
        if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
            echo "切换到分支 $BRANCH..."
            if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
                git checkout "$BRANCH"
            else
                echo "分支 $BRANCH 不存在，尝试从远程拉取..."
                git fetch "$REMOTE_NAME" "$BRANCH"
                git checkout -b "$BRANCH" "$REMOTE_NAME/$BRANCH" || git checkout -b "$BRANCH"
            fi
        fi
        
        # 设置上游并推送
        echo ""
        echo "设置上游并推送..."
        git push -u "$REMOTE_NAME" "$BRANCH"
        
        echo ""
        echo "✅ 已切换到远程仓库 $REMOTE_NAME"
        echo "   当前分支: $BRANCH"
        echo "   上游分支: $REMOTE_NAME/$BRANCH"
        ;;
        
    push)
        if [ -z "$2" ]; then
            echo "用法: $0 push <远程名称> [分支名]"
            echo "示例: $0 push gitee master"
            exit 1
        fi
        REMOTE_NAME="$2"
        BRANCH="${3:-$(git branch --show-current)}"
        
        if ! git remote get-url "$REMOTE_NAME" &> /dev/null; then
            echo "错误: 远程仓库 $REMOTE_NAME 不存在"
            exit 1
        fi
        
        echo "推送分支 $BRANCH 到远程仓库 $REMOTE_NAME..."
        git push "$REMOTE_NAME" "$BRANCH"
        echo "✅ 推送成功!"
        ;;
        
    pull)
        if [ -z "$2" ]; then
            echo "用法: $0 pull <远程名称> [分支名]"
            echo "示例: $0 pull gitee master"
            exit 1
        fi
        REMOTE_NAME="$2"
        BRANCH="${3:-$(git branch --show-current)}"
        
        if ! git remote get-url "$REMOTE_NAME" &> /dev/null; then
            echo "错误: 远程仓库 $REMOTE_NAME 不存在"
            exit 1
        fi
        
        echo "从远程仓库 $REMOTE_NAME 拉取分支 $BRANCH..."
        git pull "$REMOTE_NAME" "$BRANCH"
        echo "✅ 拉取成功!"
        ;;
        
    remove)
        if [ -z "$2" ]; then
            echo "用法: $0 remove <远程名称>"
            echo "示例: $0 remove old_repo"
            exit 1
        fi
        REMOTE_NAME="$2"
        
        if ! git remote get-url "$REMOTE_NAME" &> /dev/null; then
            echo "错误: 远程仓库 $REMOTE_NAME 不存在"
            exit 1
        fi
        
        echo "删除远程仓库 $REMOTE_NAME..."
        git remote remove "$REMOTE_NAME"
        echo "✅ 远程仓库 $REMOTE_NAME 已删除"
        ;;
        
    *)
        echo "未知命令: $1"
        echo ""
        usage
        ;;
esac

# ---------- IDE 辅助提示 ----------
if [ "$1" != "list" ] && [ "$1" != "--help" ]; then
    echo ""
    echo "========== IDE 提示 =========="
    echo "✓ .git 目录存在，IDE 将正确识别 Git 仓库"
    echo "⚠️  确保在 IDE 中打开的是仓库根目录: $(pwd)"
fi