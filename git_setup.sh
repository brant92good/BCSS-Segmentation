#!/bin/bash
# Git 快速設置與提交腳本

echo "🔧 Git 快速設置與提交"
echo "======================="
echo ""

# 檢查 git 是否已初始化
if [ ! -d ".git" ]; then
    echo "📦 初始化 Git repository..."
    git init
fi

# 配置 git (如果還沒配置)
if [ -z "$(git config user.name)" ]; then
    echo "⚙️  配置 Git user..."
    read -p "Enter your name: " git_name
    read -p "Enter your email: " git_email
    git config user.name "$git_name"
    git config user.email "$git_email"
fi

echo ""
echo "📋 當前狀態:"
echo "─────────────────────"
git status --short
echo ""

# 添加所有文件
echo "➕ 添加文件到 Git..."
git add .

echo ""
echo "📝 準備提交的文件:"
echo "─────────────────────"
git status --short
echo ""

# 顯示統計
echo "📊 統計:"
echo "─────────────────────"
echo "新增文件: $(git diff --cached --numstat | wc -l)"
echo "修改文件: $(git diff --cached --name-only | wc -l)"
echo ""

# 提交
read -p "提交訊息 (或按 Enter 使用預設): " commit_msg

if [ -z "$commit_msg" ]; then
    commit_msg="Initial commit - BCSS segmentation project with improvements"
fi

echo ""
echo "💾 提交中..."
git commit -m "$commit_msg"

echo ""
echo "✅ Git 設置完成!"
echo ""
echo "📌 下一步:"
echo "  • 檢查提交: git log --oneline"
echo "  • 添加遠端: git remote add origin <repository_url>"
echo "  • 推送: git push -u origin main"
echo ""
