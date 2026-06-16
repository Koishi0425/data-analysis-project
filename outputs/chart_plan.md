# YCSI 可视化图表清单

| 图表 | 对应研究问题 | 主要字段 | 输出文件 |
|---|---|---|---|
| YCSI 综合排名与五维贡献图 | 哪些城市综合表现更强，各维度如何贡献最终得分 | YCSI, opportunity, housing friendliness, commute friendliness, life, growth | 01_ycsi_ranking.png |
| 15城青年城市生存力空间分布 | 15 个目标城市的综合表现与空间分布 | city, lng, lat, YCSI, population_wan | 02_ycsi_city_heat_points.png |
| 城市平均月薪与住房负担关系 | 高薪是否足以抵消高房租 | avg_salary, avg_rent, rent_income_ratio, job_count | 03_salary_rent_bubble.png |
| 机会指数与居住友好度四象限 | 哪些城市在就业机会和住房负担之间更均衡 | opportunity_score, housing_friendliness, YCSI | 04_opportunity_pressure_quadrant.png |
| 聚类代表城市五维雷达图 | 不同城市类型的典型画像差异 | five YCSI dimensions, cluster representative cities | 05_city_radar.png |
| 生活便利综合得分点图 | 哪些城市的人均生活便利资源更充分 | life_score, poi_per_10k | 06a_life_score_dot.png |
| 各类 POI 构成 100% 堆积图 | 不同城市生活设施结构是否均衡 | subway, hospital, park, mall, restaurant, library, gym | 06b_poi_composition_100pct.png |
| 原始变量相关性热力图 | 薪资、租金、岗位、人均资源等原始变量之间的关系 | raw variables only | 07a_raw_variable_correlation_heatmap.png |
| 五个一级指标相关性热力图 | 五个一级指标之间是否存在结构性相关 | five YCSI dimensions only | 07b_dimension_correlation_heatmap.png |
| 五维 K-Means 城市聚类 PCA 图 | 推荐不同类型青年适合的城市类别 | five YCSI dimensions, PCA components | 08_city_clusters.png |
| 商品房租金绝对水平与指数趋势 | 各城市租金水平和相对涨跌趋势如何变化 | date, city, rent_per_sqm, rent index 2018=100 | 09_rent_trend.png |

说明：本轮为静态图优先版本，不引入外部地图底图和交互式选择器；指标方向统一为越高越好。
