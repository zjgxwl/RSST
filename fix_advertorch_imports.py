"""
批量替换所有文件中的 advertorch 导入为自定义实现
"""
import os
import re

# 需要替换的文件列表
files_to_fix = [
    'calculate_flops_regroup.py',
    'custome_tensorRT.py',
    'end-to-end.py',
    'main.py',
    'main_eval.py',
    'main_eval_fillback.py',
    'main_eval_regroup.py',
    'main_imp.py',
    'main_imp_regroup.py',
    'main_imp_skipzero.py',
    'model.py',
    'models/densenet.py',
    'models/resnet.py',
    'models/resnet50_cfg.py',
    'models/resnets.py',
    'models/resnets_2fc.py',
    'models/resnet_cfg.py',
    'models/shufflenet.py',
    'models/resnet_grasp.py',
    'models/vgg.py',
    'sparse_main_imp_fillback.py',
]

old_import = 'from advertorch.utils import NormalizeByChannelMeanStd'
new_import = '# from advertorch.utils import NormalizeByChannelMeanStd\nfrom normalize_utils import NormalizeByChannelMeanStd  # 自定义实现，功能相同'

fixed_count = 0
for file_path in files_to_fix:
    if not os.path.exists(file_path):
        print(f'跳过（文件不存在）: {file_path}')
        continue
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_import in content:
            new_content = content.replace(old_import, new_import)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f'✓ 已修复: {file_path}')
            fixed_count += 1
        else:
            print(f'跳过（未找到导入）: {file_path}')
    
    except Exception as e:
        print(f'✗ 错误 {file_path}: {e}')

print(f'\n总共修复了 {fixed_count} 个文件')
print('现在可以运行测试了！')

