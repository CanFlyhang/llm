@echo off

REM 设置远程仓库地址
git remote add origin https://github.com/CanFlyhang/llm.git

REM 提交所有文件
git add .
git commit -m "Initial commit"

REM 推送到远程仓库
git push -u origin master

echo 项目上传完成！
pause
