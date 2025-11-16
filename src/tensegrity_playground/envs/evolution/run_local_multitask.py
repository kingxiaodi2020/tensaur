# run_local_multitask.py
from tensegrity_playground.envs.evolution.single_trainer import SingleIndividualTrainer
from tensegrity_playground.envs.evolution.individual import Individual

if __name__ == "__main__":
    # 准备一个简单的基因；字段按你项目里的默认/回退来
    ind = Individual(
        individual_id="debug001",
        genes={
            "num_segments": 3,
            "segment_spacing": 0.06,
            "stiffness": 5000,
            "damping": 10,
        },
    )

    trainer = SingleIndividualTrainer()
    # 关键：multi_objective，会串行跑 slope / forward / leap 三个子任务
    res = trainer.train_individual(ind, mode="multi_objective")
    print("\n==== RESULT ====")
    print(res)
