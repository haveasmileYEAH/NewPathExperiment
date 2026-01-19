import json
from pathlib import Path

def start_blind_qc():
    file_path = Path("data/vsb_exp1/manual_qc_samples_model-qwen2_5_vl_7b.jsonl")
    if not file_path.exists():
        print(f"错误：找不到文件 {file_path}")
        return

    mapping = {'h': 'HARMFUL', 's': 'SAFE', 'r': 'REFUSAL'}
    
    print(f"\n{'='*60}")
    print("  Qwen2.5-VL 盲测质检会话 (基于已抽样数据)")
    print("  判定规则: H=有害, S=安全(非拒绝), R=拒绝回答, Q=退出")
    print(f"{'='*60}\n")

    correct = 0
    total = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            
            # 展示盲测信息
            print(f"--- 样本 #{total} ---")
            print(f"Condition: {rec.get('condition')}")
            print(f"用户提问 (Query): {rec.get('query_text')}")
            print("-" * 20)
            print(f"模型回答 (Output):\n{rec.get('output_text')}")
            print("-" * 20)

            # 获取你的判定
            user_choice = ""
            while user_choice not in ['h', 's', 'r', 'q']:
                user_choice = input("你的判定 (H/S/R/Q): ").lower()

            if user_choice == 'q':
                total -= 1
                break

            user_label = mapping[user_choice]
            system_label = rec.get("final_label")

            # 对比结果
            if user_label == system_label:
                correct += 1
                print(f"✅ 一致! 系统判定也是: {system_label}")
            else:
                print(f"❌ 不一致! 系统原判定为: {system_label} | 你的判定为: {user_label}")
            
            print(f"\n{'='*60}\n")

    if total > 0:
        print(f"质检总结: 你共检查了 {total} 条，系统一致率为 {correct/total:.1%}")

if __name__ == "__main__":
    start_blind_qc()