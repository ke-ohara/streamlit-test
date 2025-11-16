import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import math

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import pydeck as pdk  # pydeck を使う

st.set_page_config(page_title="収集ルート最適化ダッシュボード", layout="wide")
st.title("収集ルート最適化ダッシュボード（東京駅周辺・時間制約付き）")

# ================================
#  1. 距離計算系（ハーヴァーシン）
# ================================
def haversine_km(lat1, lon1, lat2, lon2):
    """2点間の球面距離[km]（直線距離）"""
    R = 6371.0  # 地球半径[km]
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ================================
#  2. データモデル作成
# ================================
def create_base_nodes():
    """車庫＋20拠点のマスタ（名前・東京駅周辺の座標・基準収集量）を作成"""
    nodes = []

    # 東京駅付近を基準
    base_lat = 35.681  # 東京駅あたり
    base_lng = 139.767

    # 0: 車庫（東京駅近く）
    nodes.append({
        "node_id": 0,
        "name": "車庫（東京駅近く）",
        "lat": base_lat,
        "lng": base_lng,
        "base_weight": 0,
    })

    # 1〜20: 収集拠点（東京駅周辺に5×4グリッドで配置）
    for i in range(1, 21):
        row = (i - 1) // 5  # 0〜3
        col = (i - 1) % 5   # 0〜4
        lat = base_lat + row * 0.002  # 約200m刻み
        lng = base_lng + col * 0.002

        nodes.append({
            "node_id": i,
            "name": f"拠点{i:02d}",
            "lat": lat,
            "lng": lng,
            "base_weight": 60 + i * 5,
        })

    return pd.DataFrame(nodes)


def build_time_matrix_from_coords(nodes_df: pd.DataFrame, avg_speed_kmph: float = 30.0):
    """
    緯度経度から距離[km]→移動時間[分]を計算して time_matrix を作る。
    実運用ではここを地図APIなどに差し替えればOK。
    """
    num_nodes = len(nodes_df)
    matrix = []
    for i in range(num_nodes):
        row = []
        lat1, lon1 = nodes_df.loc[i, "lat"], nodes_df.loc[i, "lng"]
        for j in range(num_nodes):
            if i == j:
                row.append(0)
            else:
                lat2, lon2 = nodes_df.loc[j, "lat"], nodes_df.loc[j, "lng"]
                dist_km = haversine_km(lat1, lon1, lat2, lon2)
                # 移動時間（分）= 距離(km) / 速度(km/h) * 60
                time_min = dist_km / avg_speed_kmph * 60.0
                row.append(int(round(time_min)))
        matrix.append(row)
    return matrix


def create_history(nodes_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """
    各拠点ごとの過去履歴（重さ・所要時間）をダミーで作成。
    実務ではここを実績データから作るイメージ。
    """
    records = []
    today = datetime.today().date()
    random.seed(0)

    for d in range(days):
        date = today - timedelta(days=days - 1 - d)
        for _, row in nodes_df.iterrows():
            node_id = row["node_id"]
            if node_id == 0:
                continue
            base = row["base_weight"]
            weight = max(10, int(random.gauss(mu=base, sigma=base * 0.2)))
            duration = max(3.0, 5.0 + weight / 40.0 + random.gauss(0, 2.0))
            records.append({
                "date": date,
                "node_id": node_id,
                "weight_kg": weight,
                "duration_min": round(duration, 1),
            })
    return pd.DataFrame(records)


# ================================
#  3. OR-Tools でルート計算（重量＋時間制約）
# ================================
def solve_vrp(time_matrix, demands, vehicle_capacities, service_times, max_work_time, depot=0):
    """
    time_matrix: ノード間の移動時間[分]
    demands:     各ノードの荷物重量
    service_times: 各ノードでの作業時間[分]
    max_work_time: 1台あたりの最大稼働時間[分]
    """
    data = {
        "time_matrix": time_matrix,
        "demands": demands,
        "vehicle_capacities": vehicle_capacities,
        "num_vehicles": len(vehicle_capacities),
        "depot": depot,
        "service_times": service_times,
        "max_work_time": max_work_time,
    }

    manager = pywrapcp.RoutingIndexManager(
        len(data["time_matrix"]),
        data["num_vehicles"],
        data["depot"],
    )
    routing = pywrapcp.RoutingModel(manager)

    # ---- 時間（移動＋作業）コールバック ----
    def time_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        travel = data["time_matrix"][f][t]
        service = data["service_times"][f]  # 出発ノードでの作業時間をここに乗せる
        return travel + service

    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # ---- 重量制約 ----
    def demand_callback(from_index):
        f = manager.IndexToNode(from_index)
        return data["demands"][f]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,
        data["vehicle_capacities"],
        True,
        "Capacity",
    )

    # ---- 時間制約（最大稼働時間）----
    routing.AddDimension(
        transit_idx,
        0,  # 待機時間などのスラックは今回は0
        data["max_work_time"],
        True,  # 出発時の累積時間を0に固定
        "Time",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(1)

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None

    results = []
    time_dimension = routing.GetDimensionOrDie("Time")

    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)
        route = []
        load = 0
        prev_idx = None

        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            load += data["demands"][node]

            # この地点までの累積稼働時間（移動＋作業）
            time_val = solution.Value(time_dimension.CumulVar(idx))

            route.append({
                "stop_order": len(route),
                "node_id": node,
                "累積荷重_kg": load,
                "ここまでの稼働時間_分": time_val,
            })

            prev_idx = idx
            idx = solution.Value(routing.NextVar(idx))

        # 終了地点（車庫戻り）での累積時間
        end_time = solution.Value(time_dimension.CumulVar(idx))

        results.append({
            "vehicle_id": v,
            "route": route,
            "total_time": end_time,
            "final_load": load,
        })

    return results


# ================================
#  4. データ準備
# ================================
nodes_df = create_base_nodes()
time_matrix = build_time_matrix_from_coords(nodes_df, avg_speed_kmph=30.0)
history_df = create_history(nodes_df, days=30)

# ================================
#  5. サイドバー：条件入力
# ================================
with st.sidebar:
    st.header("条件設定")
    num_vehicles = st.number_input("トラック台数", min_value=1, max_value=10, value=3, step=1)
    capacity_per_vehicle = st.number_input("1台あたりの積載容量(kg)", min_value=100, max_value=5000, value=800, step=50)
    avg_speed = st.slider("平均速度（km/h）※距離→時間の参考", min_value=10, max_value=80, value=30, step=5)

    st.markdown("---")
    base_service_time = st.number_input("1拠点あたりの基準作業時間(分)", min_value=0, max_value=120, value=10, step=5)
    time_capacity_per_vehicle = st.number_input("1台あたりの最大稼働時間(分)", min_value=60, max_value=1440, value=480, step=30)

    st.caption("※ 稼働時間 = 移動時間 + 各拠点での作業時間（基準×件数）で計算します。")

time_matrix = build_time_matrix_from_coords(nodes_df, avg_speed_kmph=float(avg_speed))

# ================================
#  6. 今日の想定重量を編集
# ================================
st.markdown("## 今日の想定収集重量")

avg_weights = (
    history_df.groupby("node_id")["weight_kg"].mean().round().astype(int)
    .reindex(nodes_df["node_id"]).fillna(0)
)

plan_df = nodes_df.copy()
plan_df["平均重量_kg"] = avg_weights.values
plan_df["今日の想定重量_kg"] = plan_df["平均重量_kg"]
plan_df.loc[plan_df["node_id"] == 0, ["平均重量_kg", "今日の想定重量_kg"]] = 0

st.info("右側の「今日の想定重量_kg」を拠点ごとに編集してから、下のボタンでルート最適化を実行できます。")

edited_plan_df = st.data_editor(
    plan_df[["node_id", "name", "平均重量_kg", "今日の想定重量_kg"]],
    num_rows="fixed",
    use_container_width=True,
)

today_demands = (
    edited_plan_df.sort_values("node_id")["今日の想定重量_kg"]
    .astype(int)
    .tolist()
)

vehicle_capacities = [int(capacity_per_vehicle)] * int(num_vehicles)

# ★ 各拠点での作業時間（ここでは「基準分」を単純に足すだけ）
#   将来的には重量に応じて増やすなどの拡張もOK
service_times = [0] * len(nodes_df)  # 車庫は0
for idx, row in nodes_df.iterrows():
    node_id = row["node_id"]
    if node_id == 0:
        service_times[node_id] = 0
    else:
        service_times[node_id] = int(base_service_time)

# ================================
#  7. ルート最適化 実行
# ================================
st.markdown("## 収集ルートの最適化")

run_opt = st.button("この条件でルートを最適化する")

results = None
if run_opt:
    with st.spinner("最適ルートを計算中..."):
        results = solve_vrp(
            time_matrix=time_matrix,
            demands=today_demands,
            vehicle_capacities=vehicle_capacities,
            service_times=service_times,
            max_work_time=int(time_capacity_per_vehicle),
            depot=0,
        )

if results is None and run_opt:
    st.error("解が見つかりませんでした（容量・台数・最大稼働時間を見直してみてください）。")

# ================================
#  8. ダッシュボード表示
# ================================
if results is not None:
    # ---- 1段目：トラック別カード（横並び） ----
    st.markdown("### トラック別ルート（カード表示）")

    cols_truck = st.columns(len(results))

    total_time_all = 0
    total_load_all = 0

    for col, res in zip(cols_truck, results):
        with col:
            bg = "linear-gradient(135deg, #f0f4ff, #ffffff)"
            st.markdown(
                f"""
                <div style="
                    padding: 16px;
                    border-radius: 12px;
                    background: {bg};
                    border: 1px solid #dce3f5;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                    margin-bottom: 12px;
                ">
                    <h3 style="margin-bottom:8px;">トラック {res['vehicle_id']}</h3>
                    <p style="margin:4px 0;">総稼働時間: <b>{res['total_time']} 分</b></p>
                    <p style="margin:4px 0;">最終累積荷重: <b>{res['final_load']} kg</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 詳細ルートテーブル
            route_records = []
            for r in res["route"]:
                node_id = r["node_id"]
                node_name = nodes_df.loc[nodes_df["node_id"] == node_id, "name"].iloc[0]
                weight = today_demands[node_id]
                route_records.append({
                    "順番": r["stop_order"],
                    "node_id": node_id,
                    "場所": node_name,
                    "この拠点の想定重量_kg": weight,
                    "累積荷重_kg": r["累積荷重_kg"],
                    "ここまでの稼働時間_分": r["ここまでの稼働時間_分"],
                })

            route_df = pd.DataFrame(route_records)
            st.dataframe(route_df, use_container_width=True, height=260)

            # 荷物増加の推移
            st.caption("荷物増加の推移（累積荷重）")
            st.line_chart(route_df.set_index("順番")["累積荷重_kg"])

            # 稼働時間の推移
            st.caption("稼働時間の推移（移動＋作業）")
            st.line_chart(route_df.set_index("順番")["ここまでの稼働時間_分"])

        total_time_all += res["total_time"]
        total_load_all += res["final_load"]

    # ---- 2段目：全体サマリカード ----
    st.markdown("### 全体サマリ（時間 / 重量）")

    st.markdown(
        f"""
        <div style="
            padding: 18px;
            border-radius: 12px;
            background: #ffffff;
            border: 1px solid #e1e4ea;
            box-shadow: 0 4px 10px rgba(0,0,0,0.04);
            margin-bottom: 16px;
        ">
            <h3>全体サマリ</h3>
            <p style="margin:4px 0;">総稼働時間: <b>{total_time_all} 分</b></p>
            <p style="margin:4px 0;">総収集量（想定）: <b>{total_load_all} kg</b></p>
            <p style="margin:4px 0;">トラック台数: <b>{len(results)} 台</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- 3段目：拠点タイル（重さ推移＋簡易地図） ----
    st.markdown("### 拠点タイル（重さ推移＋地図）")

    target_nodes = [1, 2, 3]
    cols_nodes = st.columns(len(target_nodes))

    for col, nid in zip(cols_nodes, target_nodes):
        with col:
            node_row = nodes_df.loc[nodes_df["node_id"] == nid].iloc[0]
            name = node_row["name"]

            st.markdown(
                f"""
                <div style="
                    padding: 12px;
                    border-radius: 10px;
                    background-color: #fafbff;
                    border: 1px solid #dde3f5;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
                    margin-bottom: 8px;
                ">
                    <h4 style="margin-bottom:4px;">{name}</h4>
                    <p style="margin:2px 0;">node_id: {nid}</p>
                    <p style="margin:2px 0;">座標: ({node_row['lat']:.4f}, {node_row['lng']:.4f})</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            hist_target = history_df[history_df["node_id"] == nid].sort_values("date")

            t1, t2 = st.tabs(["重さ推移", "位置（地図）"])
            with t1:
                st.line_chart(hist_target.set_index("date")["weight_kg"])
            with t2:
                st.map(
                    pd.DataFrame(
                        [{"lat": node_row["lat"], "lon": node_row["lng"]}]
                    ),
                    zoom=14,
                )

    # ---- 4段目：pydeck でルート＋拠点マップ ----
    st.markdown("### ルート全体マップ（pydeck）")

    points_df = nodes_df.copy().rename(columns={"lng": "lon"})
    points_df["type"] = points_df["node_id"].apply(lambda x: "depot" if x == 0 else "stop")

    def color_for_type(t):
        if t == "depot":
            return [255, 0, 0]   # 車庫：赤
        return [0, 122, 255]     # 拠点：青

    points_df["color"] = points_df["type"].apply(color_for_type)

    path_data = []
    palette = [
        [255, 99, 132],
        [54, 162, 235],
        [255, 206, 86],
        [75, 192, 192],
        [153, 102, 255],
    ]

    for res in results:
        vehicle_id = res["vehicle_id"]
        node_ids = [r["node_id"] for r in res["route"]]
        full_nodes = [0] + node_ids + [0]  # 車庫スタート＆帰着

        path_coords = []
        for nid in full_nodes:
            row = nodes_df.loc[nodes_df["node_id"] == nid].iloc[0]
            path_coords.append([row["lng"], row["lat"]])  # [lon, lat]

        color = palette[vehicle_id % len(palette)]

        path_data.append({
            "vehicle_id": vehicle_id,
            "path": path_coords,
            "color": color,
        })

    path_df = pd.DataFrame(path_data)

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=80,
        pickable=True,
    )

    path_layer = pdk.Layer(
        "PathLayer",
        data=path_df,
        get_path="path",
        get_color="color",
        width_scale=10,
        width_min_pixels=2,
        rounded=True,
    )

    mid_lat = nodes_df["lat"].mean()
    mid_lon = nodes_df["lng"].mean()

    view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=13,
        pitch=40,
    )

    deck = pdk.Deck(
        layers=[scatter_layer, path_layer],
        initial_view_state=view_state,
        tooltip={"text": "vehicle_id: {vehicle_id}"},
    )

    st.pydeck_chart(deck)
