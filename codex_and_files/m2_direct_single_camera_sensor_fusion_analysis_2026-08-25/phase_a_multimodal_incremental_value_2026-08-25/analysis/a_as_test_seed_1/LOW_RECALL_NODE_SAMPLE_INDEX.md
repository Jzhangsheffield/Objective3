# Low-Recall Node 样本索引

> 范围：A_as_test、seed_1；低 Recall 定义为 `< 80%`。每个样本均标注该方法下是否正确以及错误时的预测类别。

## A0 — 主相机 M2-Direct

### node_1_unlock_crimper — Recall 66.67% (4/6)

误分类样本：

- `sample_000051` → `node_4_turn_on_crimper`
- `sample_000130` → `node_35_lock_crimper`

正确分类样本：

- `sample_000005`
- `sample_000204`
- `sample_000274`
- `sample_000379`

### node_8_turn_on_extractor_fan — Recall 66.67% (4/6)

误分类样本：

- `sample_000134` → `node_28_turn_off_extractor_fan`
- `sample_000278` → `node_6_turn_on_air_compressor`

正确分类样本：

- `sample_000004`
- `sample_000054`
- `sample_000208`
- `sample_000383`

### node_24_put_sample_on_table — Recall 41.67% (10/24)

误分类样本：

- `sample_000038` → `node_12_take_plier_from_table`
- `sample_000073` → `node_12_take_plier_from_table`
- `sample_000087` → `node_12_take_plier_from_table`
- `sample_000150` → `node_12_take_plier_from_table`
- `sample_000178` → `node_12_take_plier_from_table`
- `sample_000226` → `node_34_take_lock_from_table`
- `sample_000232` → `node_12_take_plier_from_table`
- `sample_000238` → `node_12_take_plier_from_table`
- `sample_000259` → `node_12_take_plier_from_table`
- `sample_000265` → `node_12_take_plier_from_table`
- `sample_000309` → `node_12_take_plier_from_table`
- `sample_000315` → `node_12_take_plier_from_table`
- `sample_000329` → `node_12_take_plier_from_table`
- `sample_000430` → `node_12_take_plier_from_table`

正确分类样本：

- `sample_000024`
- `sample_000101`
- `sample_000115`
- `sample_000164`
- `sample_000202`
- `sample_000296`
- `sample_000353`
- `sample_000367`
- `sample_000402`
- `sample_000416`

### node_30_turn_off_air_compressor — Recall 33.33% (2/6)

误分类样本：

- `sample_000041` → `node_6_turn_on_air_compressor`
- `sample_000118` → `node_6_turn_on_air_compressor`
- `sample_000269` → `node_6_turn_on_air_compressor`
- `sample_000336` → `node_6_turn_on_air_compressor`

正确分类样本：

- `sample_000184`
- `sample_000371`

## A1 — 第二相机单独 M2-Direct

### node_1_unlock_crimper — Recall 66.67% (4/6)

误分类样本：

- `sample_000051` → `node_4_turn_on_crimper`
- `sample_000130` → `node_4_turn_on_crimper`

正确分类样本：

- `sample_000005`
- `sample_000204`
- `sample_000274`
- `sample_000379`

### node_5_adjust_parameters — Recall 66.67% (4/6)

误分类样本：

- `sample_000137` → `node_6_turn_on_air_compressor`
- `sample_000214` → `node_17_grip_sample_from_machine_table_2`

正确分类样本：

- `sample_000011`
- `sample_000060`
- `sample_000284`
- `sample_000389`

### node_24_put_sample_on_table — Recall 79.17% (19/24)

误分类样本：

- `sample_000150` → `node_2_put_lock_on_table`
- `sample_000202` → `node_12_take_plier_from_table`
- `sample_000226` → `node_2_put_lock_on_table`
- `sample_000232` → `node_2_put_lock_on_table`
- `sample_000315` → `node_34_take_lock_from_table`

正确分类样本：

- `sample_000024`
- `sample_000038`
- `sample_000073`
- `sample_000087`
- `sample_000101`
- `sample_000115`
- `sample_000164`
- `sample_000178`
- `sample_000238`
- `sample_000259`
- `sample_000265`
- `sample_000296`
- `sample_000309`
- `sample_000329`
- `sample_000353`
- `sample_000367`
- `sample_000402`
- `sample_000416`
- `sample_000430`

### node_25_put_plier_on_table — Recall 79.17% (19/24)

误分类样本：

- `sample_000203` → `node_12_take_plier_from_table`
- `sample_000227` → `node_12_take_plier_from_table`
- `sample_000266` → `node_12_take_plier_from_table`
- `sample_000368` → `node_12_take_plier_from_table`
- `sample_000431` → `node_12_take_plier_from_table`

正确分类样本：

- `sample_000025`
- `sample_000039`
- `sample_000074`
- `sample_000088`
- `sample_000102`
- `sample_000116`
- `sample_000151`
- `sample_000165`
- `sample_000179`
- `sample_000233`
- `sample_000239`
- `sample_000260`
- `sample_000297`
- `sample_000310`
- `sample_000316`
- `sample_000330`
- `sample_000354`
- `sample_000403`
- `sample_000417`

## A2 — 双相机 0.5/0.5 概率后融合

### node_1_unlock_crimper — Recall 66.67% (4/6)

误分类样本：

- `sample_000051` → `node_4_turn_on_crimper`
- `sample_000130` → `node_35_lock_crimper`

正确分类样本：

- `sample_000005`
- `sample_000204`
- `sample_000274`
- `sample_000379`

### node_24_put_sample_on_table — Recall 66.67% (16/24)

误分类样本：

- `sample_000073` → `node_12_take_plier_from_table`
- `sample_000150` → `node_2_put_lock_on_table`
- `sample_000226` → `node_2_put_lock_on_table`
- `sample_000232` → `node_12_take_plier_from_table`
- `sample_000238` → `node_12_take_plier_from_table`
- `sample_000259` → `node_12_take_plier_from_table`
- `sample_000309` → `node_12_take_plier_from_table`
- `sample_000315` → `node_34_take_lock_from_table`

正确分类样本：

- `sample_000024`
- `sample_000038`
- `sample_000087`
- `sample_000101`
- `sample_000115`
- `sample_000164`
- `sample_000178`
- `sample_000202`
- `sample_000265`
- `sample_000296`
- `sample_000329`
- `sample_000353`
- `sample_000367`
- `sample_000402`
- `sample_000416`
- `sample_000430`

## A3 — 双相机 gated residual/cross-view

### node_1_unlock_crimper — Recall 66.67% (4/6)

误分类样本：

- `sample_000051` → `node_4_turn_on_crimper`
- `sample_000130` → `node_35_lock_crimper`

正确分类样本：

- `sample_000005`
- `sample_000204`
- `sample_000274`
- `sample_000379`

### node_8_turn_on_extractor_fan — Recall 66.67% (4/6)

误分类样本：

- `sample_000134` → `node_28_turn_off_extractor_fan`
- `sample_000278` → `node_6_turn_on_air_compressor`

正确分类样本：

- `sample_000004`
- `sample_000054`
- `sample_000208`
- `sample_000383`

### node_24_put_sample_on_table — Recall 41.67% (10/24)

误分类样本：

- `sample_000038` → `node_12_take_plier_from_table`
- `sample_000073` → `node_12_take_plier_from_table`
- `sample_000087` → `node_12_take_plier_from_table`
- `sample_000150` → `node_12_take_plier_from_table`
- `sample_000178` → `node_12_take_plier_from_table`
- `sample_000226` → `node_12_take_plier_from_table`
- `sample_000232` → `node_12_take_plier_from_table`
- `sample_000238` → `node_12_take_plier_from_table`
- `sample_000259` → `node_12_take_plier_from_table`
- `sample_000265` → `node_12_take_plier_from_table`
- `sample_000309` → `node_12_take_plier_from_table`
- `sample_000315` → `node_12_take_plier_from_table`
- `sample_000329` → `node_12_take_plier_from_table`
- `sample_000430` → `node_12_take_plier_from_table`

正确分类样本：

- `sample_000024`
- `sample_000101`
- `sample_000115`
- `sample_000164`
- `sample_000202`
- `sample_000296`
- `sample_000353`
- `sample_000367`
- `sample_000402`
- `sample_000416`

### node_30_turn_off_air_compressor — Recall 50.00% (3/6)

误分类样本：

- `sample_000041` → `node_6_turn_on_air_compressor`
- `sample_000118` → `node_6_turn_on_air_compressor`
- `sample_000336` → `node_6_turn_on_air_compressor`

正确分类样本：

- `sample_000184`
- `sample_000269`
- `sample_000371`

## A4 — 主相机 + 右手 IMU

### node_1_unlock_crimper — Recall 66.67% (4/6)

误分类样本：

- `sample_000051` → `node_4_turn_on_crimper`
- `sample_000130` → `node_35_lock_crimper`

正确分类样本：

- `sample_000005`
- `sample_000204`
- `sample_000274`
- `sample_000379`

### node_8_turn_on_extractor_fan — Recall 66.67% (4/6)

误分类样本：

- `sample_000134` → `node_28_turn_off_extractor_fan`
- `sample_000278` → `node_6_turn_on_air_compressor`

正确分类样本：

- `sample_000004`
- `sample_000054`
- `sample_000208`
- `sample_000383`

### node_24_put_sample_on_table — Recall 41.67% (10/24)

误分类样本：

- `sample_000038` → `node_12_take_plier_from_table`
- `sample_000073` → `node_12_take_plier_from_table`
- `sample_000087` → `node_12_take_plier_from_table`
- `sample_000150` → `node_12_take_plier_from_table`
- `sample_000178` → `node_12_take_plier_from_table`
- `sample_000226` → `node_34_take_lock_from_table`
- `sample_000232` → `node_12_take_plier_from_table`
- `sample_000238` → `node_12_take_plier_from_table`
- `sample_000259` → `node_12_take_plier_from_table`
- `sample_000265` → `node_12_take_plier_from_table`
- `sample_000309` → `node_12_take_plier_from_table`
- `sample_000315` → `node_12_take_plier_from_table`
- `sample_000329` → `node_12_take_plier_from_table`
- `sample_000430` → `node_12_take_plier_from_table`

正确分类样本：

- `sample_000024`
- `sample_000101`
- `sample_000115`
- `sample_000164`
- `sample_000202`
- `sample_000296`
- `sample_000353`
- `sample_000367`
- `sample_000402`
- `sample_000416`

### node_30_turn_off_air_compressor — Recall 50.00% (3/6)

误分类样本：

- `sample_000041` → `node_6_turn_on_air_compressor`
- `sample_000118` → `node_6_turn_on_air_compressor`
- `sample_000336` → `node_6_turn_on_air_compressor`

正确分类样本：

- `sample_000184`
- `sample_000269`
- `sample_000371`

## A5 — 主相机 + 右手 EMG

### node_1_unlock_crimper — Recall 66.67% (4/6)

误分类样本：

- `sample_000051` → `node_4_turn_on_crimper`
- `sample_000130` → `node_35_lock_crimper`

正确分类样本：

- `sample_000005`
- `sample_000204`
- `sample_000274`
- `sample_000379`

### node_8_turn_on_extractor_fan — Recall 66.67% (4/6)

误分类样本：

- `sample_000134` → `node_28_turn_off_extractor_fan`
- `sample_000278` → `node_6_turn_on_air_compressor`

正确分类样本：

- `sample_000004`
- `sample_000054`
- `sample_000208`
- `sample_000383`

### node_24_put_sample_on_table — Recall 41.67% (10/24)

误分类样本：

- `sample_000038` → `node_12_take_plier_from_table`
- `sample_000073` → `node_12_take_plier_from_table`
- `sample_000087` → `node_12_take_plier_from_table`
- `sample_000150` → `node_12_take_plier_from_table`
- `sample_000178` → `node_12_take_plier_from_table`
- `sample_000226` → `node_12_take_plier_from_table`
- `sample_000232` → `node_12_take_plier_from_table`
- `sample_000238` → `node_12_take_plier_from_table`
- `sample_000259` → `node_12_take_plier_from_table`
- `sample_000265` → `node_12_take_plier_from_table`
- `sample_000309` → `node_12_take_plier_from_table`
- `sample_000315` → `node_12_take_plier_from_table`
- `sample_000329` → `node_12_take_plier_from_table`
- `sample_000430` → `node_12_take_plier_from_table`

正确分类样本：

- `sample_000024`
- `sample_000101`
- `sample_000115`
- `sample_000164`
- `sample_000202`
- `sample_000296`
- `sample_000353`
- `sample_000367`
- `sample_000402`
- `sample_000416`

### node_30_turn_off_air_compressor — Recall 66.67% (4/6)

误分类样本：

- `sample_000118` → `node_6_turn_on_air_compressor`
- `sample_000336` → `node_6_turn_on_air_compressor`

正确分类样本：

- `sample_000041`
- `sample_000184`
- `sample_000269`
- `sample_000371`

## A6 — 主相机 + 右手 EMG + IMU

### node_1_unlock_crimper — Recall 66.67% (4/6)

误分类样本：

- `sample_000051` → `node_4_turn_on_crimper`
- `sample_000130` → `node_35_lock_crimper`

正确分类样本：

- `sample_000005`
- `sample_000204`
- `sample_000274`
- `sample_000379`

### node_24_put_sample_on_table — Recall 41.67% (10/24)

误分类样本：

- `sample_000038` → `node_12_take_plier_from_table`
- `sample_000073` → `node_12_take_plier_from_table`
- `sample_000087` → `node_12_take_plier_from_table`
- `sample_000150` → `node_12_take_plier_from_table`
- `sample_000178` → `node_12_take_plier_from_table`
- `sample_000226` → `node_34_take_lock_from_table`
- `sample_000232` → `node_12_take_plier_from_table`
- `sample_000238` → `node_12_take_plier_from_table`
- `sample_000259` → `node_12_take_plier_from_table`
- `sample_000265` → `node_12_take_plier_from_table`
- `sample_000309` → `node_12_take_plier_from_table`
- `sample_000315` → `node_12_take_plier_from_table`
- `sample_000329` → `node_12_take_plier_from_table`
- `sample_000430` → `node_12_take_plier_from_table`

正确分类样本：

- `sample_000024`
- `sample_000101`
- `sample_000115`
- `sample_000164`
- `sample_000202`
- `sample_000296`
- `sample_000353`
- `sample_000367`
- `sample_000402`
- `sample_000416`

### node_30_turn_off_air_compressor — Recall 33.33% (2/6)

误分类样本：

- `sample_000041` → `node_6_turn_on_air_compressor`
- `sample_000118` → `node_6_turn_on_air_compressor`
- `sample_000269` → `node_6_turn_on_air_compressor`
- `sample_000336` → `node_6_turn_on_air_compressor`

正确分类样本：

- `sample_000184`
- `sample_000371`

