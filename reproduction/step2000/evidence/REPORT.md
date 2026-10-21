# Step 2000 BF16 本地评测

500/500 局完成，成功 216/500（43.2%）。

|设置|成功/局数|成功率|
|---|---:|---:|
|demo_clean|127/250|50.8%|
|demo_randomized|89/250|35.6%|

评测及补跑墙钟耗时：6.16 小时。BF16，chunk50，compile=False，不录像。

## Base 配对比较

仅比较双方完成且 seed、instruction、initial_hash、setup_hash、policy_seed 完全相同的 295 局：step2000 143/295（48.47%）；base 30/295（10.17%）。这不是500局全量base比较。两次模型使用的 normalization 来源不同，不能将差异完全归因于微调。

## 逐任务结果

|任务|clean /5|randomized /5|总体成功率|
|---|---:|---:|---:|
|adjust_bottle|5|2|70%|
|beat_block_hammer|3|2|50%|
|blocks_ranking_rgb|2|2|40%|
|blocks_ranking_size|2|2|40%|
|click_alarmclock|2|1|30%|
|click_bell|5|5|100%|
|dump_bin_bigbin|3|2|50%|
|grab_roller|5|4|90%|
|handover_block|2|0|20%|
|handover_mic|1|1|20%|
|hanging_mug|0|1|10%|
|lift_pot|0|0|0%|
|move_can_pot|1|0|10%|
|move_pillbottle_pad|0|0|0%|
|move_playingcard_away|5|4|90%|
|move_stapler_pad|1|0|10%|
|open_laptop|5|1|60%|
|open_microwave|1|3|40%|
|pick_diverse_bottles|2|1|30%|
|pick_dual_bottles|4|3|70%|
|place_a2b_left|3|3|60%|
|place_a2b_right|4|3|70%|
|place_bread_basket|2|0|20%|
|place_bread_skillet|2|2|40%|
|place_burger_fries|4|3|70%|
|place_can_basket|0|1|10%|
|place_cans_plasticbox|4|1|50%|
|place_container_plate|5|4|90%|
|place_dual_shoes|1|1|20%|
|place_empty_cup|3|2|50%|
|place_fan|0|0|0%|
|place_mouse_pad|1|1|20%|
|place_object_basket|2|1|30%|
|place_object_scale|3|1|40%|
|place_object_stand|5|2|70%|
|place_phone_stand|2|1|30%|
|place_shoe|4|3|70%|
|press_stapler|5|5|100%|
|put_bottles_dustbin|1|1|20%|
|put_object_cabinet|1|0|10%|
|rotate_qrcode|4|1|50%|
|scan_object|3|2|50%|
|shake_bottle|5|5|100%|
|shake_bottle_horizontally|5|5|100%|
|stack_blocks_three|1|0|10%|
|stack_blocks_two|2|3|50%|
|stack_bowls_three|2|0|20%|
|stack_bowls_two|1|1|20%|
|stamp_seal|0|2|20%|
|turn_switch|3|1|40%|

## 异常与范围

- 10 局达到专家预检查上限后仍执行策略，未增加专家调用；这10局成功1局，不保证专家可解。
- 2 局（dump_bin_bigbin、put_bottles_dustbin 的 randomized index0）同 seed/config 的当前初始场景与历史哈希不同；两次独立初始化验证当前哈希稳定，保留历史场景清单后执行当前场景。这2局均失败，排除严格配对比较。
- dump_bin_bigbin、place_phone_stand 出现场景初始化不稳定；错误发生在策略开始前，单独记录，按已尝试 seed 的原顺序做有界初始化恢复。完成的策略案例均未重跑。
- 其余488局成功215局（44.06%）；全500局包含上述12局，整体指标是本地评测结果，并非正式比赛提交。
- checkpoint 已完成分片/元数据核验，导出1708个张量，逐张量数值回读校验、键和形状校验通过；严格加载及真实观测完整rollout通过。
- 未找到此次训练运行独立保存的配置。推理配置来自交付的 train_8xA100.yaml 模板及 clean_norm/robotwin.json；权重架构兼容已验证，但实际训练时使用的 normalization 无法独立确证。

## 文件

- 远端结果：/root/lingbot_challenge/results/step2000_bf16/
- 单局结果：bf16/{setting}/{task}/{index}.json
- 当前/历史场景：cases/ 与 historical_scene_manifests/
- 转换校验：export_verified.json；配置来源：inference_config_provenance.json
- 推理权重：/root/deliverables/trained_ckpt/global_step_2000/hf_ckpt
- 原始checkpoint及base旧结果保留。所有本任务评测及服务器进程已退出。
