def print_iteration_results(results):
    print("\n" + "=" * 60)
    print("Noisy Student 迭代训练结果对比")
    print("=" * 60)
    for i, acc in enumerate(results):
        if i == 0:
            print(f"教师模型 (基线):       {acc:.2f}%")
        else:
            print(f"学生模型 (迭代 {i}):    {acc:.2f}%")
    if len(results) > 1:
        improvement = results[-1] - results[0]
        print(f"总提升:                +{improvement:.2f}%")
    print("=" * 60)
