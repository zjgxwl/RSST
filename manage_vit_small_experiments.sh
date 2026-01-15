#!/bin/bash

# ViT-Small实验管理脚本
# 用于查看状态、停止实验等操作

function show_status() {
    echo "=========================================="
    echo "ViT-Small实验运行状态"
    echo "=========================================="
    echo ""
    
    # 查找所有相关进程
    PROCESSES=$(ps aux | grep "vit_small" | grep "main_imp_fillback" | grep -v grep)
    
    if [ -z "$PROCESSES" ]; then
        echo "❌ 没有正在运行的实验"
    else
        echo "✅ 正在运行的实验:"
        echo ""
        echo "$PROCESSES" | while read line; do
            PID=$(echo $line | awk '{print $2}')
            CMD=$(echo $line | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}')
            
            # 提取数据集和方法
            DATASET=$(echo $CMD | grep -oP '(?<=--dataset )\w+' | head -1)
            STRUCT=$(echo $CMD | grep -oP '(?<=--struct )\w+' | head -1)
            
            echo "  PID $PID: $DATASET + $STRUCT"
        done
        
        echo ""
        echo "总进程数: $(echo "$PROCESSES" | wc -l)"
    fi
    
    echo ""
    echo "=========================================="
    
    # 显示最新日志
    echo ""
    echo "最新日志文件:"
    ls -lt logs_vit_small/*.log 2>/dev/null | head -4 | awk '{print "  " $9 " (" $6" "$7" "$8")"}'
    
    echo ""
}

function show_logs() {
    echo "=========================================="
    echo "实验日志文件"
    echo "=========================================="
    echo ""
    
    if [ ! -d "logs_vit_small" ]; then
        echo "❌ logs_vit_small目录不存在"
        return
    fi
    
    LOG_COUNT=$(ls logs_vit_small/*.log 2>/dev/null | wc -l)
    
    if [ $LOG_COUNT -eq 0 ]; then
        echo "❌ 没有找到日志文件"
    else
        echo "找到 $LOG_COUNT 个日志文件:"
        echo ""
        ls -lht logs_vit_small/*.log | awk '{print "  " NR ". " $9 " (" $5 ", " $6" "$7" "$8")"}'
    fi
    
    echo ""
}

function tail_log() {
    echo "=========================================="
    echo "选择要查看的日志"
    echo "=========================================="
    echo ""
    
    # 列出最新的日志
    LOG_FILES=($(ls -t logs_vit_small/*.log 2>/dev/null))
    
    if [ ${#LOG_FILES[@]} -eq 0 ]; then
        echo "❌ 没有找到日志文件"
        return
    fi
    
    echo "最新的日志文件:"
    for i in "${!LOG_FILES[@]}"; do
        echo "  $((i+1)). ${LOG_FILES[$i]}"
    done
    
    echo ""
    echo -n "请输入编号 (1-${#LOG_FILES[@]}): "
    read choice
    
    if [ $choice -ge 1 ] && [ $choice -le ${#LOG_FILES[@]} ]; then
        LOG_FILE="${LOG_FILES[$((choice-1))]}"
        echo ""
        echo "实时显示: $LOG_FILE"
        echo "按 Ctrl+C 退出"
        echo ""
        tail -f "$LOG_FILE"
    else
        echo "❌ 无效的选择"
    fi
}

function stop_experiments() {
    echo "=========================================="
    echo "停止实验"
    echo "=========================================="
    echo ""
    
    PROCESSES=$(ps aux | grep "vit_small" | grep "main_imp_fillback" | grep -v grep)
    
    if [ -z "$PROCESSES" ]; then
        echo "❌ 没有正在运行的实验"
        return
    fi
    
    echo "正在运行的实验:"
    echo ""
    echo "$PROCESSES" | while read line; do
        PID=$(echo $line | awk '{print $2}')
        CMD=$(echo $line | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}')
        DATASET=$(echo $CMD | grep -oP '(?<=--dataset )\w+' | head -1)
        STRUCT=$(echo $CMD | grep -oP '(?<=--struct )\w+' | head -1)
        echo "  PID $PID: $DATASET + $STRUCT"
    done
    
    echo ""
    echo -n "确认要停止所有实验吗？(yes/no): "
    read confirm
    
    if [ "$confirm" == "yes" ]; then
        echo ""
        echo "正在停止实验..."
        echo "$PROCESSES" | awk '{print $2}' | while read pid; do
            kill $pid 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "  ✓ 已停止 PID $pid"
            else
                echo "  ✗ 无法停止 PID $pid"
            fi
        done
        echo ""
        echo "✅ 所有实验已停止"
    else
        echo "❌ 操作已取消"
    fi
    
    echo ""
}

function show_results() {
    echo "=========================================="
    echo "实验结果概览"
    echo "=========================================="
    echo ""
    
    # 查找所有输出目录
    OUTPUT_DIRS=$(find output -maxdepth 1 -type d -name "vit_small_*" 2>/dev/null)
    
    if [ -z "$OUTPUT_DIRS" ]; then
        echo "❌ 没有找到实验结果"
        return
    fi
    
    echo "找到以下实验结果:"
    echo ""
    
    for dir in $OUTPUT_DIRS; do
        BASENAME=$(basename $dir)
        BEST_MODEL=$(find $dir -name "*model_SA_best.pth.tar" 2>/dev/null | head -1)
        
        echo "📁 $BASENAME"
        
        if [ -n "$BEST_MODEL" ]; then
            echo "   ✓ 找到最佳模型: $(basename $BEST_MODEL)"
            
            # 尝试从日志中提取最佳准确率
            LOG_FILE=$(ls logs_vit_small/*$(echo $BASENAME | grep -oP '(cifar\d+)_(refill|rsst)')*.log 2>/dev/null | tail -1)
            if [ -n "$LOG_FILE" ]; then
                BEST_ACC=$(grep "best SA=" $LOG_FILE 2>/dev/null | tail -1 | grep -oP '\d+\.\d+')
                if [ -n "$BEST_ACC" ]; then
                    echo "   ✓ 最佳准确率: $BEST_ACC%"
                fi
            fi
        else
            echo "   ⚠ 未找到最佳模型（可能还在训练中）"
        fi
        
        echo ""
    done
}

function show_gpu_usage() {
    echo "=========================================="
    echo "GPU使用情况"
    echo "=========================================="
    echo ""
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
        while IFS=, read -r idx name mem_used mem_total util temp; do
            echo "GPU $idx: $name"
            echo "  内存: $mem_used MB / $mem_total MB"
            echo "  利用率: $util%"
            echo "  温度: ${temp}°C"
            echo ""
        done
        
        # 显示哪些进程在使用GPU
        echo "GPU进程:"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | \
        while IFS=, read -r pid name mem; do
            # 检查是否是我们的实验
            if ps -p $pid -o cmd= | grep -q "vit_small"; then
                DATASET=$(ps -p $pid -o cmd= | grep -oP '(?<=--dataset )\w+' | head -1)
                STRUCT=$(ps -p $pid -o cmd= | grep -oP '(?<=--struct )\w+' | head -1)
                echo "  PID $pid ($DATASET+$STRUCT): $mem"
            fi
        done
    else
        echo "❌ nvidia-smi 不可用"
    fi
    
    echo ""
}

# 主菜单
function show_menu() {
    echo ""
    echo "=========================================="
    echo "ViT-Small实验管理工具"
    echo "=========================================="
    echo ""
    echo "1. 查看实验状态"
    echo "2. 查看日志列表"
    echo "3. 实时查看日志"
    echo "4. 停止所有实验"
    echo "5. 查看实验结果"
    echo "6. 查看GPU使用情况"
    echo "0. 退出"
    echo ""
    echo -n "请选择操作 (0-6): "
}

# 主循环
while true; do
    show_menu
    read choice
    
    case $choice in
        1)
            show_status
            ;;
        2)
            show_logs
            ;;
        3)
            tail_log
            ;;
        4)
            stop_experiments
            ;;
        5)
            show_results
            ;;
        6)
            show_gpu_usage
            ;;
        0)
            echo ""
            echo "👋 再见！"
            echo ""
            exit 0
            ;;
        *)
            echo ""
            echo "❌ 无效的选择，请重试"
            ;;
    esac
    
    echo ""
    echo -n "按回车继续..."
    read
done
