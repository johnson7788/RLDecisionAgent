# 命令行参数
https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html
命令行参数的介绍会分为基本参数，原子参数、集成参数和特定模型参数。命令行最终使用的参数列表为集成参数。集成参数继承自基本参数和一些原子参数。特定模型参数是针对于具体模型的参数，可以通过--model_kwargs'或者环境变量进行设置。
命令行传入list使用空格隔开即可。例如：--dataset <dataset_path1> <dataset_path2>。
命令行传入dict使用json。例如：--model_kwargs '{"fps_max_frames": 12}'。



# 如何debug：
你可以使用以下方式进行debug，这与使用命令行微调是等价的，但此方式不支持分布式。微调命令行运行入口可以查看这里。
from swift.llm import sft_main, TrainArguments
result = sft_main(TrainArguments(
    model='Qwen/Qwen2.5-7B-Instruct',
    train_type='lora',
    dataset=['AI-ModelScope/alpaca-gpt4-data-zh#500',
             'AI-ModelScope/alpaca-gpt4-data-en#500',
             'swift/self-cognition#500'],
    torch_dtype='bfloat16',
    # ...
))


