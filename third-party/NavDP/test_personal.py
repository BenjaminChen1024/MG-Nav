import habitat_sim

# 场景文件路径，请替换为你的 HM3D 场景文件
# 如果你没有，可以使用 habitat-sim 自带的测试场景
# "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
# 这个场景文件在你安装 habitat-sim 时通常会下载。
SCENE_PATH = "你的HM3d场景文件路径.glb"

def test_scene_navigation(scene_path):
    """加载一个场景并测试其导航网格。"""

    # 创建一个空的配置对象
    settings = habitat_sim.AgentConfiguration()

    # 创建一个模拟器配置
    cfg = habitat_sim.Configuration(
        sim_cfg=habitat_sim.SimulatorConfiguration(
            scene_id=scene_path,
            enable_physics=False,
        ),
        agents_cfg=[settings],
    )

    try:
        # 创建并初始化模拟器
        sim = habitat_sim.Simulator(cfg)

        # 获取导航网格路径寻找器
        pathfinder = sim.pathfinder

        # 检查导航网格是否有效
        if not pathfinder.is_navigable(habitat_sim.Vector3(0, 0, 0)):
            print("警告: 场景的导航网格似乎不包含原点 (0, 0, 0)。")

        # 尝试获取一个随机可导航点
        # 这里就是你之前出错的地方
        random_point = pathfinder.get_random_navigable_point()

        print(f"成功在场景中找到了一个可导航点: {random_point}")

    except Exception as e:
        print(f"发生错误: {e}")
        print("这通常意味着导航网格有问题，可能场景文件损坏，或者无法生成有效的导航区域。")

    finally:
        # 确保模拟器被正确关闭
        if 'sim' in locals():
            sim.close()

if __name__ == "__main__":
    # 替换为你的 HM3D 场景文件路径
    test_scene_navigation("/nas_dataset/wangbo/HM3D/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb")

    # 尝试用 habitat 自带的场景测试，确保你的代码本身没有问题
    print("\n--- 正在用 habitat-sim 自带的测试场景进行验证 ---")
    test_scene_navigation("/nas_dataset/wangbo/navigation_data/scene_datasets/mp3d/ZMojNkEp431/ZMojNkEp431.glb")