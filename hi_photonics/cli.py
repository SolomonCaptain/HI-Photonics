"""
HI-Photonics 命令行接口
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        prog="hi-photonics",
        description="HI-Photonics: 光子学逆向设计框架"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # version 命令
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="显示版本信息"
    )
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="列出可用资源")
    list_parser.add_argument(
        "type",
        choices=["challenges", "simulators", "models"],
        help="资源类型"
    )
    
    # run 命令
    run_parser = subparsers.add_parser("run", help="运行逆向设计")
    run_parser.add_argument(
        "-c", "--challenge",
        default="grating_coupler",
        help="挑战名称 (default: grating_coupler)"
    )
    run_parser.add_argument(
        "-m", "--model",
        default="hilab",
        choices=["tnn", "mdn", "cgan", "pinn", "hilab"],
        help="模型类型 (default: hilab)"
    )
    run_parser.add_argument(
        "-t", "--target",
        type=str,
        help="目标性能 (JSON 格式)"
    )
    run_parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=50,
        help="优化迭代次数 (default: 50)"
    )
    run_parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=100,
        help="训练轮数 (default: 100)"
    )
    run_parser.add_argument(
        "-o", "--output",
        default="outputs",
        help="输出目录 (default: outputs)"
    )
    run_parser.add_argument(
        "--no-sim",
        action="store_true",
        help="跳过仿真验证"
    )
    
    # api 命令
    api_parser = subparsers.add_parser("api", help="启动 API 服务")
    api_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址 (default: 0.0.0.0)"
    )
    api_parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="端口 (default: 8000)"
    )
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="开发模式（自动重载）"
    )
    
    # train 命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "-c", "--challenge",
        required=True,
        help="挑战名称"
    )
    train_parser.add_argument(
        "-m", "--model",
        required=True,
        choices=["tnn", "mdn", "cgan", "pinn", "hilab"],
        help="模型类型"
    )
    train_parser.add_argument(
        "-d", "--data",
        help="训练数据路径 (HDF5)"
    )
    train_parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=1000,
        help="合成数据样本数 (default: 1000)"
    )
    train_parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=100,
        help="训练轮数 (default: 100)"
    )
    train_parser.add_argument(
        "-o", "--output",
        default="models/saved",
        help="模型保存路径 (default: models/saved)"
    )
    
    args = parser.parse_args()
    
    # 显示版本
    if args.version:
        from hi_photonics import get_version
        print(f"hi-photonics version {get_version()}")
        return 0
    
    # 处理命令
    if args.command == "list":
        handle_list(args)
    elif args.command == "run":
        handle_run(args)
    elif args.command == "api":
        handle_api(args)
    elif args.command == "train":
        handle_train(args)
    else:
        parser.print_help()
        return 1
    
    return 0


def handle_list(args):
    """处理 list 命令"""
    from hi_photonics import list_available_challenges, list_available_simulators
    
    if args.type == "challenges":
        challenges = list_available_challenges()
        print("可用挑战:")
        for c in challenges:
            print(f"  - {c}")
    
    elif args.type == "simulators":
        simulators = list_available_simulators()
        print("可用仿真器:")
        for s in simulators:
            print(f"  - {s}")
    
    elif args.type == "models":
        print("可用模型:")
        for m in ["tnn", "mdn", "cgan", "pinn", "hilab"]:
            print(f"  - {m}")


def handle_run(args):
    """处理 run 命令"""
    from hi_photonics import create_pipeline, PipelineConfig
    
    print(f"运行逆向设计...")
    print(f"  挑战: {args.challenge}")
    print(f"  模型: {args.model}")
    
    # 解析目标性能
    target = None
    if args.target:
        try:
            target = json.loads(args.target)
        except json.JSONDecodeError:
            print(f"错误: 无效的 JSON 目标性能: {args.target}")
            return 1
    
    # 创建管道
    config = PipelineConfig(
        name=f"cli_run_{args.challenge}",
        challenge_name=args.challenge,
        model_type=args.model,
        target_performance=target,
        num_iterations=args.iterations,
        num_epochs=args.epochs,
        output_dir=args.output,
        use_simulation=not args.no_sim,
        verbose=True
    )
    
    pipeline = create_pipeline(
        challenge_name=args.challenge,
        model_type=args.model,
        num_iterations=args.iterations,
        num_epochs=args.epochs,
        output_dir=args.output,
        use_simulation=not args.no_sim
    )
    
    # 运行
    try:
        result = pipeline.run()
        print("\n设计完成!")
        
        if 'design_path' in result:
            print(f"  设计保存至: {result['design_path']}")
        if 'summary_path' in result:
            print(f"  摘要保存至: {result['summary_path']}")
        
        return 0
    except Exception as e:
        print(f"错误: {e}")
        return 1


def handle_api(args):
    """处理 api 命令"""
    import uvicorn
    
    print(f"启动 API 服务...")
    print(f"  地址: {args.host}:{args.port}")
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def handle_train(args):
    """处理 train 命令"""
    from hi_photonics import (
        create_pipeline, PipelineConfig,
        SyntheticDataset, create_dataloaders,
        ChallengeFactory
    )
    from hi_photonics import create_hilab_for_challenge, create_tnn_for_challenge
    
    print(f"训练模型...")
    print(f"  挑战: {args.challenge}")
    print(f"  模型: {args.model}")
    
    # 获取挑战
    challenge = ChallengeFactory.create(args.challenge)
    design_shape = challenge.spec.get_grid_shape()
    
    # 创建数据
    if args.data:
        from hi_photonics import HDF5Dataset
        dataset = HDF5Dataset(args.data)
    else:
        print(f"  生成合成数据: {args.num_samples} 样本")
        dataset = SyntheticDataset(
            num_samples=args.num_samples,
            design_shape=design_shape,
            performance_dim=challenge.spec.performance_dim
        )
    
    train_loader, val_loader, _ = create_dataloaders(dataset)
    
    # 创建模型
    if args.model == "hilab":
        model = create_hilab_for_challenge(
            args.challenge,
            latent_dim=32,
            performance_dim=challenge.spec.performance_dim
        )
    elif args.model == "tnn":
        model = create_tnn_for_challenge(args.challenge)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # 训练
    print(f"  开始训练: {args.epochs} 轮")
    
    if args.model == "hilab":
        model.train_vae(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs
        )
    else:
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs
        )
    
    # 保存模型
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / f"{args.model}_{args.challenge}.pt"
    if hasattr(model, 'save'):
        model.save(model_path)
    else:
        import torch
        torch.save(model.state_dict(), model_path)
    
    print(f"  模型保存至: {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
