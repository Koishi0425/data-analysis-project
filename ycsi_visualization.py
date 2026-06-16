from __future__ import annotations

import os
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_YEAR = 2021

CITY_COORDS = {
    "北京": (116.4074, 39.9042),
    "上海": (121.4737, 31.2304),
    "广州": (113.2644, 23.1291),
    "深圳": (114.0579, 22.5431),
    "杭州": (120.1551, 30.2741),
    "南京": (118.7969, 32.0603),
    "苏州": (120.5853, 31.2989),
    "成都": (104.0665, 30.5728),
    "重庆": (106.5516, 29.5630),
    "武汉": (114.3054, 30.5931),
    "西安": (108.9398, 34.3416),
    "长沙": (112.9388, 28.2282),
    "郑州": (113.6254, 34.7466),
    "天津": (117.2000, 39.1333),
    "青岛": (120.3826, 36.0671),
}

COORDS_BY_ORDER = [
    (116.4074, 39.9042),
    (121.4737, 31.2304),
    (113.2644, 23.1291),
    (114.0579, 22.5431),
    (120.1551, 30.2741),
    (118.7969, 32.0603),
    (120.5853, 31.2989),
    (104.0665, 30.5728),
    (106.5516, 29.5630),
    (114.3054, 30.5931),
    (108.9398, 34.3416),
    (112.9388, 28.2282),
    (113.6254, 34.7466),
    (117.2000, 39.1333),
    (120.3826, 36.0671),
]

YANGTZE_DELTA_CITIES = ["上海", "苏州", "杭州", "南京"]

POI_COLS = ["subway", "hospital", "park", "mall", "restaurant", "library", "gym"]
POI_DISPLAY = {
    "subway": "地铁",
    "hospital": "医院",
    "park": "公园",
    "mall": "商场",
    "restaurant": "餐饮",
    "library": "图书馆",
    "gym": "健身房",
}
POI_WEIGHTS = pd.Series(
    {
        "subway_per_10k": 0.18,
        "hospital_per_10k": 0.16,
        "park_per_10k": 0.14,
        "mall_per_10k": 0.14,
        "restaurant_per_10k": 0.16,
        "library_per_10k": 0.10,
        "gym_per_10k": 0.12,
    }
)

DIMENSION_COLS = {
    "机会": "opportunity_score",
    "居住友好": "housing_friendliness",
    "通勤友好": "commute_friendliness",
    "生活便利": "life_score",
    "成长潜力": "growth_score",
}
DIMENSION_WEIGHTS = {
    "opportunity_score": 0.30,
    "housing_friendliness": 0.20,
    "commute_friendliness": 0.10,
    "life_score": 0.20,
    "growth_score": 0.20,
}
DIMENSION_PALETTE = {
    "opportunity_score": "#2f6fb0",
    "housing_friendliness": "#3b8f5a",
    "commute_friendliness": "#2aa6a1",
    "life_score": "#16a085",
    "growth_score": "#c99a2e",
}
COLORS = {
    "ycsi": "#3f4a8a",
    "ycsi_dark": "#26346d",
    "opportunity": "#2f6fb0",
    "housing": "#3b8f5a",
    "commute": "#2aa6a1",
    "life": "#16a085",
    "growth": "#c99a2e",
    "pressure": "#d95f02",
    "muted": "#c9cdd2",
    "grid": "#d8dde3",
    "text": "#253040",
}

CHART_PLAN = [
    {
        "chart": "YCSI 综合排名与五维贡献图",
        "question": "哪些城市综合表现更强，各维度如何贡献最终得分",
        "fields": "YCSI, opportunity, housing friendliness, commute friendliness, life, growth",
        "output": "01_ycsi_ranking.png",
    },
    {
        "chart": "15城青年城市生存力空间分布",
        "question": "15 个目标城市的综合表现与空间分布",
        "fields": "city, lng, lat, YCSI, population_wan",
        "output": "02_ycsi_city_heat_points.png",
    },
    {
        "chart": "城市平均月薪与住房负担关系",
        "question": "高薪是否足以抵消高房租",
        "fields": "avg_salary, avg_rent, rent_income_ratio, job_count",
        "output": "03_salary_rent_bubble.png",
    },
    {
        "chart": "机会指数与居住友好度四象限",
        "question": "哪些城市在就业机会和住房负担之间更均衡",
        "fields": "opportunity_score, housing_friendliness, YCSI",
        "output": "04_opportunity_pressure_quadrant.png",
    },
    {
        "chart": "聚类代表城市五维雷达图",
        "question": "不同城市类型的典型画像差异",
        "fields": "five YCSI dimensions, cluster representative cities",
        "output": "05_city_radar.png",
    },
    {
        "chart": "生活便利综合得分点图",
        "question": "哪些城市的人均生活便利资源更充分",
        "fields": "life_score, poi_per_10k",
        "output": "06a_life_score_dot.png",
    },
    {
        "chart": "各类 POI 构成 100% 堆积图",
        "question": "不同城市生活设施结构是否均衡",
        "fields": "subway, hospital, park, mall, restaurant, library, gym",
        "output": "06b_poi_composition_100pct.png",
    },
    {
        "chart": "原始变量相关性热力图",
        "question": "薪资、租金、岗位、人均资源等原始变量之间的关系",
        "fields": "raw variables only",
        "output": "07a_raw_variable_correlation_heatmap.png",
    },
    {
        "chart": "五个一级指标相关性热力图",
        "question": "五个一级指标之间是否存在结构性相关",
        "fields": "five YCSI dimensions only",
        "output": "07b_dimension_correlation_heatmap.png",
    },
    {
        "chart": "五维 K-Means 城市聚类 PCA 图",
        "question": "推荐不同类型青年适合的城市类别",
        "fields": "five YCSI dimensions, PCA components",
        "output": "08_city_clusters.png",
    },
    {
        "chart": "商品房租金绝对水平与指数趋势",
        "question": "各城市租金水平和相对涨跌趋势如何变化",
        "fields": "date, city, rent_per_sqm, rent index 2018=100",
        "output": "09_rent_trend.png",
    },
]


def read_csv_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median())
    span = values.max() - values.min()
    if pd.isna(span) or span == 0:
        return pd.Series(np.full(len(values), 0.5), index=series.index)
    return (values - values.min()) / span


def score_minmax(series: pd.Series, floor: float = 0.05) -> pd.Series:
    """Min-Max for scoring: maps the sample range to [floor, 1]."""
    return floor + minmax(series) * (1 - floor)


def scale_bubble(series: pd.Series, min_size: float = 90, max_size: float = 850) -> pd.Series:
    return min_size + minmax(series) * (max_size - min_size)


def safe_per_10k(numerator: pd.Series, population_wan: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(population_wan, errors="coerce").replace(0, np.nan)
    values = pd.to_numeric(numerator, errors="coerce") / denominator
    return values.replace([np.inf, -np.inf], np.nan).fillna(values.median())


def save_figure(filename: str) -> Path:
    path = OUTPUT_DIR / filename
    fig = plt.gcf()
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def copy_compat_output(source_filename: str, target_filename: str) -> Path:
    source = OUTPUT_DIR / source_filename
    target = OUTPUT_DIR / target_filename
    target.write_bytes(source.read_bytes())
    return target


def write_chart_plan() -> Path:
    lines = [
        "# YCSI 可视化图表清单",
        "",
        "| 图表 | 对应研究问题 | 主要字段 | 输出文件 |",
        "|---|---|---|---|",
    ]
    for item in CHART_PLAN:
        lines.append(
            f"| {item['chart']} | {item['question']} | {item['fields']} | {item['output']} |"
        )
    lines.extend(
        [
            "",
            "说明：本轮为静态图优先版本，不引入外部地图底图和交互式选择器；指标方向统一为越高越好。",
            "",
        ]
    )
    path = OUTPUT_DIR / "chart_plan.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_base_data() -> pd.DataFrame:
    city_basic = read_csv_fallback(DATA_DIR / "city_basic.csv")
    poi = read_csv_fallback(DATA_DIR / "poi_data.csv")

    industry_path = PROJECT_ROOT / "city_basic_15_with_industry.csv"
    if industry_path.exists():
        industry = read_csv_fallback(industry_path)[
            [
                "city",
                "primary_industry_pct",
                "secondary_industry_pct",
                "tertiary_industry_pct",
            ]
        ]
        city_basic = city_basic.merge(industry, on="city", how="left")
    else:
        city_basic["primary_industry_pct"] = np.nan
        city_basic["secondary_industry_pct"] = np.nan
        city_basic["tertiary_industry_pct"] = np.nan

    for frame in (city_basic, poi):
        frame.columns = [str(column).strip() for column in frame.columns]

    metrics = city_basic.merge(poi, on="city", how="left")
    numeric_cols = [
        "gdp",
        "population",
        "disposable_income",
        "university_count",
        "metro_lines",
        "primary_industry_pct",
        "secondary_industry_pct",
        "tertiary_industry_pct",
        *POI_COLS,
    ]
    for column in numeric_cols:
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
            metrics[column] = metrics[column].fillna(metrics[column].median())

    coords = metrics["city"].map(CITY_COORDS)
    metrics["lng"] = coords.map(lambda item: item[0] if isinstance(item, tuple) else np.nan)
    metrics["lat"] = coords.map(lambda item: item[1] if isinstance(item, tuple) else np.nan)

    if metrics["lng"].isna().any() and len(metrics) == len(COORDS_BY_ORDER):
        ordered = pd.DataFrame(COORDS_BY_ORDER, columns=["lng_fallback", "lat_fallback"])
        metrics["lng"] = metrics["lng"].fillna(ordered["lng_fallback"])
        metrics["lat"] = metrics["lat"].fillna(ordered["lat_fallback"])

    return metrics


def add_job_and_rent_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """接入真实薪资、租金、人口和就业数据。"""
    metrics = metrics.copy()
    source_notes: list[str] = []

    snap_path = CLEAN_DIR / "city_snapshot.csv"
    if not snap_path.exists():
        raise FileNotFoundError(f"缺少清洗数据: {snap_path}，请先运行 python clean_data.py")

    snap = read_csv_fallback(snap_path)
    snap.columns = [str(c).strip() for c in snap.columns]
    cols = {
        "city": "city",
        "avg_salary_month": "avg_salary",
        "rent_month_est": "avg_rent",
        "rent_income_ratio": "rent_income_ratio",
        "urban_employment_wan": "urban_employment_wan",
        "population_wan": "population_wan",
        "gdp_per_capita": "gdp_per_capita",
        "gdp_yi": "gdp_snapshot_yi",
    }
    snap = snap[list(cols.keys())].rename(columns=cols)
    for col in [
        "avg_salary",
        "avg_rent",
        "rent_income_ratio",
        "urban_employment_wan",
        "population_wan",
        "gdp_per_capita",
        "gdp_snapshot_yi",
    ]:
        snap[col] = pd.to_numeric(snap[col], errors="coerce")

    snap["job_count"] = (snap["urban_employment_wan"] * 10000).round(0)
    metrics = metrics.merge(snap, on="city", how="left")

    for col in [
        "avg_salary",
        "avg_rent",
        "rent_income_ratio",
        "urban_employment_wan",
        "population_wan",
        "gdp_per_capita",
        "gdp_snapshot_yi",
        "job_count",
    ]:
        if metrics[col].isna().any():
            metrics[col] = metrics[col].fillna(metrics[col].median())
            source_notes.append(f"提示：{col} 存在缺失，已用中位数填充。")

    source_notes.extend(
        [
            f"薪资数据：城镇非私营单位在岗职工平均工资/12（{BASE_YEAR}，data/job_data.csv）。",
            f"租金数据：商品房平均出租价格×50㎡（{BASE_YEAR}，data/rent_data.xlsx）。",
            "人口口径：city_snapshot.csv 的 population_wan，用于人均和每万人指标。",
            "岗位规模：以城镇非私营单位从业人员数代理。",
        ]
    )
    return metrics, source_notes


def calculate_ycsi(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()

    metrics["population_for_rate_wan"] = metrics["population_wan"]
    metrics["job_per_10k"] = safe_per_10k(metrics["job_count"], metrics["population_for_rate_wan"])
    metrics["university_per_10k"] = safe_per_10k(
        metrics["university_count"], metrics["population_for_rate_wan"]
    )
    metrics["metro_line_per_10k"] = safe_per_10k(
        metrics["metro_lines"], metrics["population_for_rate_wan"]
    )
    metrics["poi_total"] = metrics[POI_COLS].sum(axis=1)
    metrics["poi_per_10k"] = safe_per_10k(metrics["poi_total"], metrics["population_for_rate_wan"])
    for col in POI_COLS:
        metrics[f"{col}_per_10k"] = safe_per_10k(
            metrics[col], metrics["population_for_rate_wan"]
        )
    metrics["subway_per_10k"] = metrics["subway_per_10k"].fillna(0)

    metrics["opportunity_score"] = (
        0.45 * score_minmax(metrics["avg_salary"])
        + 0.40 * score_minmax(metrics["job_per_10k"])
        + 0.15 * score_minmax(metrics["tertiary_industry_pct"])
    )
    metrics["housing_friendliness"] = score_minmax(1 - minmax(metrics["rent_income_ratio"]))

    commute_access = (
        0.70 * score_minmax(metrics["subway_per_10k"])
        + 0.30 * score_minmax(metrics["metro_line_per_10k"])
    )
    metrics["commute_friendliness"] = score_minmax(commute_access)

    poi_rate_cols = [f"{col}_per_10k" for col in POI_COLS]
    poi_scaled = metrics[poi_rate_cols].apply(score_minmax)
    metrics["life_score"] = score_minmax((poi_scaled * POI_WEIGHTS).sum(axis=1))

    metrics["growth_score"] = (
        0.35 * score_minmax(metrics["gdp_per_capita"])
        + 0.30 * score_minmax(metrics["disposable_income"])
        + 0.20 * score_minmax(metrics["university_per_10k"])
        + 0.15 * score_minmax(metrics["tertiary_industry_pct"])
    )

    for col, weight in DIMENSION_WEIGHTS.items():
        metrics[f"{col}_contribution"] = metrics[col] * weight * 100

    metrics["ycsi_raw"] = sum(
        metrics[col] * weight for col, weight in DIMENSION_WEIGHTS.items()
    )
    metrics["ycsi"] = (metrics["ycsi_raw"] * 100).round(2)
    metrics["rank"] = metrics["ycsi"].rank(ascending=False, method="first").astype(int)

    # 兼容部分旧输出字段名，但后续图表只使用友好度方向。
    metrics["rent_pressure"] = 1 - metrics["housing_friendliness"]
    metrics["commute_score"] = 1 - metrics["commute_friendliness"]
    metrics["rent_friendliness"] = metrics["housing_friendliness"]

    return metrics


def plot_ycsi_ranking(metrics: pd.DataFrame) -> Path:
    plot_df = metrics.sort_values("ycsi", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 8))

    contribution_cols = [f"{col}_contribution" for col in DIMENSION_COLS.values()]
    labels = list(DIMENSION_COLS.keys())
    left = np.zeros(len(plot_df))

    focus_cities = set(metrics.nsmallest(3, "rank")["city"]).union(
        set(metrics.nlargest(3, "rank")["city"])
    )
    for label, dim_col, contrib_col in zip(labels, DIMENSION_COLS.values(), contribution_cols):
        bars = ax.barh(
            plot_df["city"],
            plot_df[contrib_col],
            left=left,
            color=DIMENSION_PALETTE[dim_col],
            edgecolor="white",
            linewidth=0.6,
            label=label,
        )
        for bar, city in zip(bars, plot_df["city"]):
            bar.set_alpha(0.98 if city in focus_cities else 0.62)
        left += plot_df[contrib_col].to_numpy()

    for idx, row in plot_df.iterrows():
        color = COLORS["text"] if row["city"] in focus_cities else "#6d7680"
        weight = "bold" if row["city"] in focus_cities else "normal"
        ax.text(
            row["ycsi"] + 0.8,
            idx,
            f"{row['ycsi']:.1f}",
            va="center",
            fontsize=9,
            color=color,
            fontweight=weight,
        )

    ax.set_xlim(0, max(100, plot_df["ycsi"].max() + 8))
    ax.set_xlabel("YCSI 综合得分（5-100，五维加权贡献）")
    ax.set_title("15 城青年城市生存力指数排名与五维贡献")
    ax.grid(axis="x", alpha=0.22, color=COLORS["grid"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(title="五维贡献", ncol=5, frameon=False, loc="lower right")
    fig.text(
        0.08,
        0.015,
        "公式：YCSI = 0.30×机会 + 0.20×居住友好 + 0.10×通勤友好 + 0.20×生活便利 + 0.20×成长潜力",
        fontsize=9,
        color="#5a6470",
    )
    return save_figure("01_ycsi_ranking.png")


def annotate_city(ax: plt.Axes, row: pd.Series, fontsize: float = 8.4) -> None:
    offsets = {
        "上海": (0.30, -0.38),
        "苏州": (0.28, 0.16),
        "杭州": (-1.10, -0.28),
        "南京": (-1.10, 0.24),
        "深圳": (0.28, 0.12),
        "广州": (-0.95, 0.34),
        "天津": (0.16, 0.20),
    }
    dx, dy = offsets.get(row["city"], (0.18, 0.14))
    ax.text(row["lng"] + dx, row["lat"] + dy, row["city"], fontsize=fontsize)


def add_size_legend(ax: plt.Axes, values: pd.Series, label: str) -> None:
    quantiles = values.quantile([0.25, 0.5, 0.75]).round(0).astype(int).tolist()
    seen: set[int] = set()
    handles = []
    for value in quantiles:
        if value in seen:
            continue
        seen.add(value)
        handles.append(
            ax.scatter(
                [],
                [],
                s=float(scale_bubble(pd.Series([values.min(), value, values.max()])).iloc[1]),
                facecolors="none",
                edgecolors="#6b7280",
                linewidth=0.8,
                label=f"{value:,}",
            )
        )
    if handles:
        legend = ax.legend(
            handles=handles,
            title=label,
            frameon=False,
            loc="lower left",
            borderpad=0.3,
            labelspacing=1.1,
        )
        ax.add_artist(legend)


def plot_city_spatial_distribution(metrics: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7.6))
    size = scale_bubble(metrics["population_wan"], 110, 760)
    scatter = ax.scatter(
        metrics["lng"],
        metrics["lat"],
        c=metrics["ycsi"],
        s=size,
        cmap="Purples",
        edgecolor="#253040",
        linewidth=0.75,
        alpha=0.88,
    )
    for _, row in metrics.iterrows():
        annotate_city(ax, row)

    ax.set_xlim(102, 123)
    ax.set_ylim(21, 41)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title("15城青年城市生存力空间分布")
    ax.grid(alpha=0.22, color=COLORS["grid"])
    color_bar = plt.colorbar(scatter, ax=ax, shrink=0.78, pad=0.02)
    color_bar.set_label("YCSI 综合得分")
    add_size_legend(ax, metrics["population_wan"], "气泡面积：人口（万人）")

    inset = inset_axes(ax, width="36%", height="36%", loc="lower right", borderpad=1.0)
    local = metrics[metrics["city"].isin(YANGTZE_DELTA_CITIES)]
    inset.scatter(
        local["lng"],
        local["lat"],
        c=local["ycsi"],
        s=scale_bubble(local["population_wan"], 120, 520),
        cmap="Purples",
        edgecolor="#253040",
        linewidth=0.7,
        alpha=0.9,
        vmin=metrics["ycsi"].min(),
        vmax=metrics["ycsi"].max(),
    )
    for _, row in local.iterrows():
        annotate_city(inset, row, fontsize=8)
    inset.set_title("长三角局部", fontsize=9)
    inset.set_xlim(118.0, 122.4)
    inset.set_ylim(29.6, 32.7)
    inset.grid(alpha=0.20, color=COLORS["grid"])
    inset.tick_params(labelsize=7)

    fig.text(
        0.10,
        0.015,
        "说明：当前版本为经纬度气泡空间分布图，未引入外部地图底图；颜色表示YCSI，气泡面积表示人口。",
        fontsize=8.8,
        color="#5a6470",
    )
    return save_figure("02_ycsi_city_heat_points.png")


def plot_salary_rent_bubble(metrics: pd.DataFrame) -> Path:
    salary_median = metrics["avg_salary"].median()
    ratio_median = metrics["rent_income_ratio"].median()

    def classify(row: pd.Series) -> str:
        high_salary = row["avg_salary"] >= salary_median
        high_burden = row["rent_income_ratio"] >= ratio_median
        if high_salary and high_burden:
            return "高薪-高负担"
        if high_salary and not high_burden:
            return "高薪-低负担"
        if not high_salary and high_burden:
            return "低薪-高负担"
        return "低薪-低负担"

    plot_df = metrics.copy()
    plot_df["salary_rent_quadrant"] = plot_df.apply(classify, axis=1)
    palette = {
        "高薪-高负担": COLORS["pressure"],
        "高薪-低负担": COLORS["housing"],
        "低薪-高负担": "#e7a66a",
        "低薪-低负担": "#8aa6c8",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    for name, group in plot_df.groupby("salary_rent_quadrant"):
        ax.scatter(
            group["avg_salary"],
            group["rent_income_ratio"],
            s=scale_bubble(group["job_count"], 130, 980),
            color=palette[name],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.8,
            label=name,
        )
        for _, row in group.iterrows():
            ax.text(
                row["avg_salary"] + 55,
                row["rent_income_ratio"] + 0.003,
                row["city"],
                fontsize=8.5,
            )

    ax.axvline(salary_median, color="#555555", linestyle="--", linewidth=1)
    ax.axhline(ratio_median, color="#555555", linestyle="--", linewidth=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("平均月薪（元/月）")
    ax.set_ylabel("住房负担：月租金 / 平均月薪")
    ax.set_title("城市平均月薪与住房负担关系")
    ax.grid(alpha=0.22, color=COLORS["grid"])
    ax.legend(title="中位数划分", frameon=False, loc="best")
    ax.text(
        0.01,
        -0.13,
        "虚线为15城中位数；气泡面积表示岗位规模代理（城镇非私营单位从业人员数），散点面积与岗位规模成正比。",
        transform=ax.transAxes,
        fontsize=8.8,
        color="#5a6470",
    )
    return save_figure("03_salary_rent_bubble.png")


def plot_opportunity_housing_quadrant(metrics: pd.DataFrame) -> Path:
    opp_median = metrics["opportunity_score"].median()
    housing_median = metrics["housing_friendliness"].median()

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.add_patch(Rectangle((0, housing_median), opp_median, 1, color="#eaf5ef", alpha=0.55, zorder=0))
    ax.add_patch(Rectangle((opp_median, housing_median), 1, 1, color="#e8f1fb", alpha=0.55, zorder=0))
    ax.add_patch(Rectangle((0, 0), opp_median, housing_median, color="#f2f4f7", alpha=0.70, zorder=0))
    ax.add_patch(Rectangle((opp_median, 0), 1, housing_median, color="#fff4e8", alpha=0.70, zorder=0))

    scatter = ax.scatter(
        metrics["opportunity_score"],
        metrics["housing_friendliness"],
        s=scale_bubble(metrics["ycsi"], 130, 720),
        c=metrics["ycsi"],
        cmap="Purples",
        edgecolor="white",
        linewidth=0.9,
        alpha=0.90,
        zorder=3,
    )
    label_offsets = {
        "重庆": (-0.030, 0.018),
        "郑州": (-0.020, 0.038),
        "长沙": (0.020, 0.027),
        "武汉": (0.016, -0.020),
        "西安": (0.015, 0.026),
        "青岛": (0.014, 0.018),
        "天津": (0.014, -0.025),
        "成都": (0.012, -0.025),
        "深圳": (0.014, -0.028),
        "北京": (0.014, 0.012),
        "上海": (0.014, 0.012),
    }
    for _, row in metrics.iterrows():
        dx, dy = label_offsets.get(row["city"], (0.012, 0.012))
        ax.text(
            row["opportunity_score"] + dx,
            row["housing_friendliness"] + dy,
            row["city"],
            fontsize=8.2,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.4},
        )

    ax.axvline(opp_median, color="#555555", linestyle="--", linewidth=1)
    ax.axhline(housing_median, color="#555555", linestyle="--", linewidth=1)
    ax.text(0.09, housing_median + 0.06, "低机会\n高居住友好", ha="center", va="center", fontsize=10, color=COLORS["housing"])
    ax.text(0.88, 0.92, "高机会\n高居住友好", ha="center", va="center", fontsize=10, color=COLORS["opportunity"])
    ax.text(0.88, 0.10, "高机会\n低居住友好", ha="center", va="center", fontsize=10, color=COLORS["pressure"])
    ax.text(
        0.02,
        -0.15,
        "结论：在所选15座城市中，没有城市同时实现顶级就业机会和极低居住压力；虚线为15城中位数。",
        transform=ax.transAxes,
        fontsize=8.8,
        color="#5a6470",
    )
    ax.set_xlabel("机会指数（越高越好）")
    ax.set_ylabel("居住友好度（越高住房负担越低）")
    ax.set_title("城市机会指数与居住友好度四象限")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.22, color=COLORS["grid"])
    color_bar = plt.colorbar(scatter, ax=ax, shrink=0.78)
    color_bar.set_label("YCSI 综合得分")
    return save_figure("04_opportunity_pressure_quadrant.png")


def plot_life_score_dot(metrics: pd.DataFrame) -> Path:
    plot_df = metrics.sort_values("life_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 7))
    ax.scatter(
        plot_df["life_score"],
        plot_df["city"],
        s=scale_bubble(plot_df["poi_per_10k"], 90, 560),
        color=COLORS["life"],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.88,
    )
    for _, row in plot_df.iterrows():
        ax.text(row["life_score"] + 0.015, row["city"], f"{row['life_score']:.2f}", va="center", fontsize=8.5)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("生活便利度综合得分（按每万人POI标准化）")
    ax.set_ylabel("城市")
    ax.set_title("生活便利综合得分点图")
    ax.grid(axis="x", alpha=0.22, color=COLORS["grid"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(
        0.08,
        0.015,
        "气泡面积表示每万人POI总量；得分由地铁、医院、公园、商场、餐饮、图书馆、健身房等人均指标加权得到。",
        fontsize=8.8,
        color="#5a6470",
    )
    return save_figure("06a_life_score_dot.png")


def plot_poi_composition_100pct(metrics: pd.DataFrame) -> Path:
    plot_df = metrics.sort_values("life_score", ascending=False).copy()
    poi_share = plot_df[POI_COLS].div(plot_df[POI_COLS].sum(axis=1), axis=0)
    poi_share.index = plot_df["city"]
    poi_share = poi_share.rename(columns=POI_DISPLAY)

    fig, ax = plt.subplots(figsize=(11, 7))
    poi_share.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c", width=0.74)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_ylabel("各类POI占比")
    ax.set_xlabel("城市")
    ax.set_title("各类 POI 构成 100% 堆积图")
    ax.grid(axis="y", alpha=0.22, color=COLORS["grid"])
    ax.legend(title="POI 类型", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.xticks(rotation=35, ha="right")
    return save_figure("06b_poi_composition_100pct.png")


def correlation_and_pvalue(frame: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(frame.columns)
    corr = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
    pvalue = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns, dtype=float)
    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns):
            if i == j:
                continue
            x = pd.to_numeric(frame[col_i], errors="coerce")
            y = pd.to_numeric(frame[col_j], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                corr.loc[col_i, col_j] = np.nan
                pvalue.loc[col_i, col_j] = np.nan
                continue
            if method == "spearman":
                result = stats.spearmanr(x[valid], y[valid])
            else:
                result = stats.pearsonr(x[valid], y[valid])
            corr.loc[col_i, col_j] = result.statistic
            pvalue.loc[col_i, col_j] = result.pvalue
    return corr, pvalue


def hclust_order(corr: pd.DataFrame) -> list[str]:
    if len(corr) <= 2:
        return list(corr.columns)
    values = corr.abs().fillna(0).to_numpy()
    np.fill_diagonal(values, 1)
    distances = np.clip(1 - values, 0, 1)
    condensed = squareform(distances, checks=False)
    cluster = linkage(condensed, method="average")
    return corr.columns[leaves_list(cluster)].tolist()


def plot_lower_triangle_heatmap(
    data: pd.DataFrame,
    labels: dict[str, str],
    title: str,
    filename: str,
    csv_prefix: str,
) -> Path:
    pearson, pearson_p = correlation_and_pvalue(data, "pearson")
    spearman, spearman_p = correlation_and_pvalue(data, "spearman")

    order = hclust_order(pearson)
    pearson = pearson.loc[order, order]
    pearson_p = pearson_p.loc[order, order]
    spearman.loc[order, order].to_csv(
        OUTPUT_DIR / f"{csv_prefix}_spearman.csv", encoding="utf-8-sig"
    )
    pearson.to_csv(OUTPUT_DIR / f"{csv_prefix}_pearson.csv", encoding="utf-8-sig")
    pearson_p.to_csv(OUTPUT_DIR / f"{csv_prefix}_pearson_pvalue.csv", encoding="utf-8-sig")
    spearman_p.loc[order, order].to_csv(
        OUTPUT_DIR / f"{csv_prefix}_spearman_pvalue.csv", encoding="utf-8-sig"
    )

    corr_values = pearson.to_numpy()
    mask = np.triu(np.ones_like(corr_values, dtype=bool))
    display = np.ma.masked_where(mask, corr_values)
    lower = corr_values[np.tril_indices_from(corr_values, k=-1)]
    finite_lower = lower[np.isfinite(lower)]
    all_positive = len(finite_lower) > 0 and finite_lower.min() >= 0
    cmap = plt.cm.YlOrRd.copy() if all_positive else plt.cm.RdBu_r.copy()
    cmap.set_bad(color="white")
    vmin, vmax = (0, 1) if all_positive else (-1, 1)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax)
    axis_labels = [labels[col] for col in order]
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(axis_labels, rotation=45, ha="right")
    ax.set_yticklabels(axis_labels)

    for row_idx in range(len(order)):
        for col_idx in range(len(order)):
            if row_idx <= col_idx:
                continue
            value = pearson.iloc[row_idx, col_idx]
            p_val = pearson_p.iloc[row_idx, col_idx]
            if pd.isna(value):
                text = ""
            elif pd.notna(p_val) and p_val >= 0.10:
                text = "×"
            else:
                text = f"{value:.2f}"
            color = "white" if abs(value) > 0.62 and text != "×" else "#253040"
            if text == "×":
                color = "#9aa1aa"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8.5, color=color)

    ax.set_title(title)
    color_bar = plt.colorbar(image, ax=ax, shrink=0.80)
    color_bar.set_label("Pearson 相关系数")
    ax.text(
        0,
        -0.16,
        "仅展示下三角；× 表示 Pearson 相关在 p<0.10 下不显著；Spearman 结果另存为 CSV。",
        transform=ax.transAxes,
        fontsize=8.8,
        color="#5a6470",
    )
    return save_figure(filename)


def plot_correlation_heatmaps(metrics: pd.DataFrame) -> list[Path]:
    raw_columns = {
        "avg_salary": "平均月薪",
        "avg_rent": "平均租金",
        "rent_income_ratio": "租金收入比",
        "job_per_10k": "每万人岗位",
        "gdp_per_capita": "人均GDP",
        "disposable_income": "人均可支配收入",
        "tertiary_industry_pct": "第三产业占比",
        "university_per_10k": "每万人高校",
        "subway_per_10k": "每万人地铁",
        "poi_per_10k": "每万人POI",
    }
    dim_columns = {
        "opportunity_score": "机会指数",
        "housing_friendliness": "居住友好",
        "commute_friendliness": "通勤友好",
        "life_score": "生活便利",
        "growth_score": "成长潜力",
    }
    return [
        plot_lower_triangle_heatmap(
            metrics[list(raw_columns.keys())],
            raw_columns,
            "原始变量相关性热力图",
            "07a_raw_variable_correlation_heatmap.png",
            "07a_raw_variable_correlation",
        ),
        plot_lower_triangle_heatmap(
            metrics[list(dim_columns.keys())],
            dim_columns,
            "五个一级指标相关性热力图",
            "07b_dimension_correlation_heatmap.png",
            "07b_dimension_correlation",
        ),
    ]


def name_cluster(center: pd.Series, center_median: pd.Series) -> str:
    high = center >= center_median
    if high["opportunity_score"] and not high["housing_friendliness"]:
        return "超级机会-高成本型"
    if high["housing_friendliness"] and high["life_score"] and not high["opportunity_score"]:
        return "生活友好-中等机会型"
    if high["housing_friendliness"] and high["growth_score"]:
        return "低成本-成长潜力型"
    if high.mean() >= 0.6:
        return "综合均衡型"
    top_feature = center.idxmax()
    return {
        "opportunity_score": "机会导向型",
        "housing_friendliness": "居住友好型",
        "commute_friendliness": "通勤友好型",
        "life_score": "生活便利型",
        "growth_score": "成长潜力型",
    }.get(top_feature, "综合过渡型")


def plot_city_clusters(metrics: pd.DataFrame) -> tuple[Path, pd.DataFrame, pd.DataFrame, list[str]]:
    feature_cols = list(DIMENSION_COLS.values())
    plot_df = metrics.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(plot_df[feature_cols])

    candidates: list[tuple[int, float, np.ndarray, KMeans]] = []
    for k in (3, 4):
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        candidates.append((k, score, labels, model))
    best_k, best_score, labels, model = max(candidates, key=lambda item: item[1])
    plot_df["cluster_id"] = labels

    centers = plot_df.groupby("cluster_id")[feature_cols].mean()
    center_median = centers.median()
    raw_names = {cluster_id: name_cluster(row, center_median) for cluster_id, row in centers.iterrows()}
    name_counts: dict[str, int] = {}
    cluster_names: dict[int, str] = {}
    for cluster_id in sorted(raw_names):
        name = raw_names[cluster_id]
        name_counts[name] = name_counts.get(name, 0) + 1
        cluster_names[cluster_id] = name if name_counts[name] == 1 else f"{name}-{name_counts[name]}"
    plot_df["cluster_name"] = plot_df["cluster_id"].map(cluster_names)

    representatives: list[str] = []
    for cluster_id, group in plot_df.groupby("cluster_id"):
        center_scaled = model.cluster_centers_[cluster_id]
        group_indices = group.index.to_numpy()
        distance = np.linalg.norm(scaled[group_indices] - center_scaled, axis=1)
        representative_idx = group_indices[int(distance.argmin())]
        representative = plot_df.loc[representative_idx, "city"]
        representatives.append(representative)
        plot_df.loc[group_indices, "cluster_representative"] = representative

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    plot_df["pca_1"] = components[:, 0]
    plot_df["pca_2"] = components[:, 1]

    fig, ax = plt.subplots(figsize=(9.7, 7.2))
    colors = plt.cm.Set2(np.linspace(0, 1, plot_df["cluster_id"].nunique()))
    for color, (name, group) in zip(colors, plot_df.groupby("cluster_name")):
        ax.scatter(
            group["pca_1"],
            group["pca_2"],
            s=scale_bubble(group["ycsi"], 120, 620),
            color=color,
            alpha=0.86,
            edgecolor="white",
            linewidth=0.9,
            label=name,
        )
        for _, row in group.iterrows():
            marker = "★" if row["city"] == row["cluster_representative"] else ""
            ax.text(row["pca_1"] + 0.04, row["pca_2"] + 0.04, f"{row['city']}{marker}", fontsize=8.5)

    ax.axhline(0, color="#d0d4da", linewidth=0.8)
    ax.axvline(0, color="#d0d4da", linewidth=0.8)
    ax.set_xlabel(f"PCA 1（解释 {pca.explained_variance_ratio_[0] * 100:.1f}%）")
    ax.set_ylabel(f"PCA 2（解释 {pca.explained_variance_ratio_[1] * 100:.1f}%）")
    ax.set_title(f"五维 K-Means 城市聚类 PCA 图（K={best_k}，轮廓系数={best_score:.2f}）")
    ax.grid(alpha=0.22, color=COLORS["grid"])
    ax.legend(title="聚类类型", frameon=False, loc="best")
    figure_path = save_figure("08_city_clusters.png")

    summary = (
        plot_df.groupby(["cluster_id", "cluster_name"])
        .agg(
            representative_city=("cluster_representative", "first"),
            city_list=("city", lambda values: "、".join(map(str, values))),
            city_count=("city", "count"),
            avg_ycsi=("ycsi", "mean"),
            opportunity_score=("opportunity_score", "mean"),
            housing_friendliness=("housing_friendliness", "mean"),
            commute_friendliness=("commute_friendliness", "mean"),
            life_score=("life_score", "mean"),
            growth_score=("growth_score", "mean"),
        )
        .round(3)
        .reset_index()
    )
    summary["selected_k"] = best_k
    summary["silhouette_score"] = round(float(best_score), 3)

    return figure_path, summary, plot_df, representatives


def plot_city_radar(metrics: pd.DataFrame, representatives: list[str]) -> Path:
    labels = list(DIMENSION_COLS.keys())
    columns = list(DIMENSION_COLS.values())
    selected = list(dict.fromkeys(representatives))
    for city in metrics.sort_values("ycsi", ascending=False)["city"]:
        if len(selected) >= 5:
            break
        if city not in selected:
            selected.append(city)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8.3, 8.3), subplot_kw={"polar": True})
    color_map = plt.cm.Set2(np.linspace(0, 1, len(selected)))
    for color, city in zip(color_map, selected):
        values = metrics.loc[metrics["city"] == city, columns].iloc[0].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=city, color=color)
        ax.fill(angles, values, alpha=0.05, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title("聚类代表城市 YCSI 五维雷达图", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.24, 1.08), frameon=False)
    return save_figure("05_city_radar.png")


def plot_rent_trend(metrics: pd.DataFrame, representative_cities: list[str]) -> Path | None:
    panel_path = CLEAN_DIR / "rent_panel.csv"
    if not panel_path.exists():
        return None
    panel = read_csv_fallback(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["rent_per_sqm"] = pd.to_numeric(panel["rent_per_sqm"], errors="coerce")
    panel = panel.dropna(subset=["date", "rent_per_sqm"])
    panel = panel[panel["city"].isin(metrics["city"])]

    base = (
        panel[panel["date"].dt.year == 2018]
        .groupby("city")["rent_per_sqm"]
        .mean()
        .rename("rent_2018")
    )
    if base.empty:
        base = panel.groupby("city")["rent_per_sqm"].first().rename("rent_2018")
    panel = panel.merge(base, on="city", how="left")
    panel["rent_index_2018_100"] = panel["rent_per_sqm"] / panel["rent_2018"] * 100

    highlight = list(dict.fromkeys(representative_cities))
    if not highlight:
        highlight = metrics.sort_values("ycsi", ascending=False)["city"].head(5).tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(highlight)))
    color_by_city = dict(zip(highlight, colors))

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for city, grp in panel.groupby("city"):
        grp = grp.sort_values("date")
        if city in color_by_city:
            axes[0].plot(grp["date"], grp["rent_per_sqm"], linewidth=2.1, label=city, color=color_by_city[city], zorder=3)
            axes[1].plot(grp["date"], grp["rent_index_2018_100"], linewidth=2.1, label=city, color=color_by_city[city], zorder=3)
        else:
            axes[0].plot(grp["date"], grp["rent_per_sqm"], linewidth=0.8, color=COLORS["muted"], alpha=0.55, zorder=1)
            axes[1].plot(grp["date"], grp["rent_index_2018_100"], linewidth=0.8, color=COLORS["muted"], alpha=0.55, zorder=1)

    axes[0].set_ylabel("元/㎡·月")
    axes[0].set_title("15 城商品房租金绝对水平")
    axes[1].axhline(100, color="#555555", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("租金指数（2018=100）")
    axes[1].set_title("15 城商品房租金指数趋势")
    axes[1].set_xlabel("时间")
    for ax in axes:
        ax.grid(alpha=0.22, color=COLORS["grid"])
    axes[0].legend(title="聚类代表城市", frameon=False, loc="upper left", ncol=3)
    axes[1].text(
        0,
        -0.25,
        "指数计算：RentIndex_it = Rent_it / Rent_i,2018 × 100；曲线为原始月度数据，未使用移动平均。",
        transform=axes[1].transAxes,
        fontsize=8.8,
        color="#5a6470",
    )
    return save_figure("09_rent_trend.png")


def export_scores(metrics: pd.DataFrame) -> Path:
    columns = [
        "city",
        "rank",
        "ycsi",
        "ycsi_raw",
        "avg_salary",
        "avg_rent",
        "rent_income_ratio",
        "job_count",
        "population_wan",
        "job_per_10k",
        "poi_per_10k",
        "university_per_10k",
        "subway_per_10k",
        "metro_line_per_10k",
        "gdp_per_capita",
        "tertiary_industry_pct",
        "opportunity_score",
        "housing_friendliness",
        "commute_friendliness",
        "life_score",
        "growth_score",
        "opportunity_score_contribution",
        "housing_friendliness_contribution",
        "commute_friendliness_contribution",
        "life_score_contribution",
        "growth_score_contribution",
    ]
    score_df = metrics.sort_values("rank")[columns].reset_index(drop=True)
    path = OUTPUT_DIR / "ycsi_city_scores.csv"
    score_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_cluster_summary(summary: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "cluster_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def validate_outputs(metrics: pd.DataFrame, generated: list[Path]) -> list[str]:
    issues: list[str] = []
    if metrics["city"].nunique() != 15:
        issues.append(f"城市数量异常：{metrics['city'].nunique()}，预期15。")
    expected_ranks = set(range(1, len(metrics) + 1))
    if set(metrics["rank"]) != expected_ranks:
        issues.append("排名不是连续的1-15。")
    if not metrics["ycsi"].between(5, 100).all():
        issues.append("YCSI 不在5-100范围内。")
    for col in list(DIMENSION_COLS.values()) + [
        "job_per_10k",
        "poi_per_10k",
        "university_per_10k",
        "subway_per_10k",
        "gdp_per_capita",
    ]:
        values = pd.to_numeric(metrics[col], errors="coerce")
        if values.isna().any() or np.isinf(values).any():
            issues.append(f"{col} 存在缺失或无穷值。")
    for path in generated:
        if not path.exists() or path.stat().st_size == 0:
            issues.append(f"输出文件为空或缺失：{path}")
    return issues


def run() -> None:
    plan_path = write_chart_plan()
    metrics = load_base_data()
    metrics, source_notes = add_job_and_rent_metrics(metrics)
    metrics = calculate_ycsi(metrics)

    cluster_path, cluster_summary, clustered_metrics, representatives = plot_city_clusters(metrics)
    metrics = metrics.merge(
        clustered_metrics[["city", "cluster_id", "cluster_name", "cluster_representative", "pca_1", "pca_2"]],
        on="city",
        how="left",
    )

    poi_paths = [
        plot_life_score_dot(metrics),
        plot_poi_composition_100pct(metrics),
    ]
    correlation_paths = plot_correlation_heatmaps(metrics)
    compat_paths = [
        copy_compat_output("06b_poi_composition_100pct.png", "06_poi_convenience_stack.png"),
        copy_compat_output("07a_raw_variable_correlation_heatmap.png", "07_metric_correlation_heatmap.png"),
    ]

    generated = [
        plan_path,
        export_scores(metrics),
        plot_ycsi_ranking(metrics),
        plot_city_spatial_distribution(metrics),
        plot_salary_rent_bubble(metrics),
        plot_opportunity_housing_quadrant(metrics),
        plot_city_radar(metrics, representatives),
        *poi_paths,
        *correlation_paths,
        *compat_paths,
        cluster_path,
        export_cluster_summary(cluster_summary),
    ]

    trend_path = plot_rent_trend(metrics, representatives)
    if trend_path is not None:
        generated.append(trend_path)

    validation_issues = validate_outputs(metrics, generated)

    print("数据来源说明：")
    for note in source_notes:
        print(f"- {note}")
    print("\nYCSI Top 5：")
    print(metrics.sort_values("rank")[["rank", "city", "ycsi"]].head(5).to_string(index=False))
    print("\n聚类代表城市：")
    print("、".join(representatives))
    print("\n已生成文件：")
    for path in generated:
        print(f"- {path}")
    if validation_issues:
        print("\n校验提示：")
        for issue in validation_issues:
            print(f"- {issue}")
    else:
        print("\n校验通过：15城齐全，排名连续，YCSI在5-100内，关键派生指标和输出文件有效。")


if __name__ == "__main__":
    run()
