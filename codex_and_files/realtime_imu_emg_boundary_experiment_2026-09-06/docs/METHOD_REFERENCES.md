# 方法依据与参考文献

本实验包的第一版结构由可穿戴传感器编码、因果时序分割和显式边界分支三部分组成。

1. Ordóñez, F. J. and Roggen, D. *Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition*. Sensors, 2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC4732148/
2. Lea, C. et al. *Temporal Convolutional Networks for Action Segmentation and Detection*. CVPR, 2017. https://openaccess.thecvf.com/content_cvpr_2017/papers/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf
3. Abu Farha, Y. and Gall, J. *MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation*. CVPR, 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html
4. Ishikawa, Y. et al. *Alleviating Over-Segmentation Errors by Detecting Action Boundaries*. WACV, 2021. https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html
5. Santuz, A. et al. *A systematic review of automatic muscle activity detection methods for surface EMG*. 2023. https://pubmed.ncbi.nlm.nih.gov/37872633/
6. Solnik, S. et al. *Teager-Kaiser energy operator signal conditioning improves EMG onset detection*. https://pmc.ncbi.nlm.nih.gov/articles/PMC5425195/
7. Phinyomark, A. et al. *Feature reduction and selection for EMG signal classification*. 关于实时 EMG 窗口长度的比较：https://www.sciencedirect.com/science/article/pii/S0208521617300323
8. Besomi, M. et al. *Consensus for experimental design in electromyography*. https://www.sciencedirect.com/science/article/pii/S1050641120300821

这些文献支持所选模型家族和预处理范围，但本数据集的 20–200 Hz EMG 上限、20 Hz 决策网格和具体 debounce 参数仍是待通过训练折 validation 验证的实验设计，不应作为已被结果证明的最优值。
