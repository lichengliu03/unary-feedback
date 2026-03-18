#!/bin/bash
# 检查实验进度脚本

echo "=========================================="
echo "实验进度监控"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 检查后台进程
echo "【后台监控进程】"
ps aux | grep "run_full_experiment.sh" | grep -v grep | awk '{print "  PID " $2 ": " $11 " " $12 " " $13}'
if [ $? -ne 0 ]; then
    echo "  无运行中的监控进程"
fi
echo ""

# 检查SLURM任务
echo "【SLURM任务队列】"
squeue -u lliu22 -o "  %-12i %-25j %-12T %-12M %-8D %R" | head -20
echo ""

# 检查日志文件
echo "【实验日志状态】"
for log in run_1turn_no_feedback.log run_5turn_no_feedback.log; do
    if [ -f "$log" ]; then
        lines=$(wc -l < "$log")
        last_update=$(stat -c %y "$log" | cut -d'.' -f1)
        echo "  $log:"
        echo "    行数: $lines"
        echo "    最后更新: $last_update"
        echo "    最后5行:"
        tail -5 "$log" | sed 's/^/      /'
        echo ""
    fi
done

# 检查checkpoint目录
echo "【Checkpoint状态】"
for exp in qwen25_3b_1turn_no_feedback qwen25_3b_5turn_no_feedback; do
    ckpt_dir="/projects/bfea/lliu22/ragen_checkpoints/${exp}"
    if [ -d "$ckpt_dir" ]; then
        echo "  ${exp}:"
        steps=$(ls -d ${ckpt_dir}/global_step_* 2>/dev/null | wc -l)
        echo "    已保存步数: $steps"
        if [ $steps -gt 0 ]; then
            ls -d ${ckpt_dir}/global_step_* 2>/dev/null | xargs -n1 basename | sed 's/^/      /'
        fi
    else
        echo "  ${exp}: 目录不存在"
    fi
    echo ""
done

echo "=========================================="
