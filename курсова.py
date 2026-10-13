import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
from scipy.ndimage import distance_transform_edt
import warnings

# distance_transform_edt обчислює евклiдову вiдстань вiд кожного
# пiкселя растра до найближчого нульового елемента (заборонена зона
# або межа дороги), що дозволяє формувати матрицю придатностi S.
warnings.filterwarnings('ignore')

# Параметри дослiджуваної територiї
KYIV_BBOX = (50.213, 50.591, 30.239, 30.825)
south, north, west, east = KYIV_BBOX

CENTER_LON, CENTER_LAT = 30.5234, 50.4501 # центр мiста Київ

GRID_H, GRID_W = 800, 800 # розмiр растрової сiтки

# Гiперпараметри оптимiзатора Adam
ALPHA = 0.01
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
T_MAX = 300 # кiлькiсть iтерацiй; обгрунтування: збiжнiсть
# досягається при t~200 (|delta_t - delta_{t-1}|
# / |delta_{t-1}| < 1e-4), запас 50% для надiйностi

# Ваги критерiїв (mu_o)
MU_TIM = 0.30
MU_GDP = 0.30
MU_COS = 0.20
MU_LIV = 0.20

GAMMA = 20.0 # параметр жорсткостi LSE-апроксимацiї

# 1. Завантаження геоданих з OpenStreetMap
print("Завантаження дорожньої мережi...")
bbox = (west, south, east, north)
G = ox.graph_from_bbox(bbox=bbox, network_type='drive')
nodes, edges = ox.graph_to_gdfs(G)

print("Завантаження шарiв землекористування...")
tags = {'landuse': ['industrial', 'residential', 'forest'],
        'natural': 'water',
        'leisure': 'park'}
land = ox.features_from_bbox(bbox=bbox, tags=tags)

# 2. Формування растрових матриць придатностi
transform = from_bounds(west, south, east, north, GRID_W, GRID_H)

def rasterize_feature(gdf, shape, transform):
    """Растеризацiя векторного шару у бiнарну матрицю."""
    from rasterio.features import rasterize
    geoms = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not geoms:
        return np.zeros(shape, dtype=np.float32)
    return rasterize(geoms, out_shape=shape,
                     transform=transform, fill=0,
                     dtype=np.float32)

# Транспортна доступнiсть: вiдстань до найближчої дороги
road_mask = rasterize_feature(edges, (GRID_H, GRID_W), transform)
S_tim_raw = distance_transform_edt(1 - road_mask)
S_tim = 1 - S_tim_raw / (S_tim_raw.max() + 1e-9) # normalizatsiya

# Економiчний критерiй: промисловi зони
industrial = land[land.get('landuse') == 'industrial'] \
    if 'landuse' in land.columns else gpd.GeoDataFrame()
ind_mask = rasterize_feature(industrial, (GRID_H, GRID_W), transform)
ind_dist = distance_transform_edt(1 - ind_mask)
S_gdp = 1 - ind_dist / (ind_dist.max() + 1e-9)

# Вартiсний критерiй: обернена вiдстань вiд центру (proxy)
lons = np.linspace(west, east, GRID_W)
lats = np.linspace(north, south, GRID_H)
LON_GRID, LAT_GRID = np.meshgrid(lons, lats)
dist_center = np.sqrt((LON_GRID - CENTER_LON)**2 + (LAT_GRID - CENTER_LAT)**2)
S_cos = dist_center / dist_center.max() # далi вiд центру = дешевше

# Екологiчний критерiй: вiдстань вiд житлової забудови
residential = land[land.get('landuse') == 'residential'] \
    if 'landuse' in land.columns else gpd.GeoDataFrame()
res_mask = rasterize_feature(residential, (GRID_H, GRID_W), transform)
res_dist = distance_transform_edt(1 - res_mask)
S_liv = res_dist / (res_dist.max() + 1e-9)

# Маска заборонених зон (вода + парки + ліси)
water = land[land.get('natural') == 'water'] \
    if 'natural' in land.columns else gpd.GeoDataFrame()
parks = land[land.get('leisure') == 'park'] \
    if 'leisure' in land.columns else gpd.GeoDataFrame()
forests = land[(land.get('natural') == 'wood') | (land.get('landuse') == 'forest')] \
    if ('natural' in land.columns) or ('landuse' in land.columns) else gpd.GeoDataFrame()

water_mask = rasterize_feature(water, (GRID_H, GRID_W), transform)
park_mask = rasterize_feature(parks, (GRID_H, GRID_W), transform)
forest_mask = rasterize_feature(forests, (GRID_H, GRID_W), transform)

S_mask = torch.tensor(
    1 - np.clip(water_mask + park_mask + forest_mask, 0, 1),
    dtype=torch.float32)

# 3. Конвертацiя у тензори PyTorch
S_tim_t = torch.tensor(S_tim, dtype=torch.float32)
S_gdp_t = torch.tensor(S_gdp, dtype=torch.float32)
S_cos_t = torch.tensor(S_cos, dtype=torch.float32)
S_liv_t = torch.tensor(S_liv, dtype=torch.float32)

# 4. Iнiцiалiзацiя параметрiв оптимiзацiї
theta = torch.zeros(GRID_H, GRID_W, requires_grad=True)
optimizer = torch.optim.Adam([theta],
                             lr=ALPHA, betas=(BETA1, BETA2),
                             eps=EPSILON)

# 5. Утопiчна точка (теоретичнi максимуми критерiїв)
f_tim_star = float(S_tim_t.max())
f_gdp_star = float(S_gdp_t.max())
f_cos_star = float(S_cos_t.max())
f_liv_star = float(S_liv_t.max())

f_tim_nadir = float(S_tim_t.min())
f_gdp_nadir = float(S_gdp_t.min())
f_cos_nadir = float(S_cos_t.min())
f_liv_nadir = float(S_liv_t.min())

def safe_norm(val, star, nadir):
    """Нормалiзацiя вiдхилення у дiапазон [0, 1]."""
    denom = abs(nadir - star) + 1e-9
    return abs(val - star) / denom

# 6. Головний цикл оптимiзацiї
print(f"Запуск оптимiзацiї Adam: {T_MAX} iтерацiй, alpha={ALPHA}")
loss_history = []

for t in range(1, T_MAX + 1):
    optimizer.zero_grad()

    # Сигмоїдне вiдображення у ймовiрнiсний простiр [0, 1]
    p = torch.sigmoid(theta) * S_mask

    # Обчислення нормалiзованих критерiїв
    mask_sum = S_mask.sum() + 1e-9
    f_tim = (p * S_tim_t).sum() / mask_sum
    f_gdp = (p * S_gdp_t).sum() / mask_sum
    f_cos = (p * S_cos_t).sum() / mask_sum
    f_liv = (p * S_liv_t).sum() / mask_sum

    # Нормалiзованi вiдхилення вiд утопiчних значень
    d_tim = safe_norm(f_tim, f_tim_star, f_tim_nadir) / MU_TIM
    d_gdp = safe_norm(f_gdp, f_gdp_star, f_gdp_nadir) / MU_GDP
    d_cos = safe_norm(f_cos, f_cos_star, f_cos_nadir) / MU_COS
    d_liv = safe_norm(f_liv, f_liv_star, f_liv_nadir) / MU_LIV

    # LSE-апроксимацiя мiнiмаксної функцiї (формула 2.12)
    scores = torch.stack([d_tim, d_gdp, d_cos, d_liv])
    scores = torch.stack([d_tim, d_gdp, d_cos, d_liv])

    # Зберігаємо максимум окремо
    max_score = scores.max()

    # Робимо чисельно стабільний зсув
    scores_shifted = scores - max_score

    loss = max_score + (1.0 / GAMMA) * torch.log(
        torch.exp(GAMMA * scores_shifted).sum()
    )
    loss.backward()

    # Проєкцiя градiєнта: обнулiння у заборонених зонах
    if theta.grad is not None:
        theta.grad.data *= S_mask

    optimizer.step()
    loss_history.append(loss.item())

    if t % 50 == 0:
        print(f" Iтерацiя {t:3d}: loss = {loss.item():.6f}")

# 7. Постпроцесинг та збереження результатiв
with torch.no_grad():
    p_final = (torch.sigmoid(theta) * S_mask).numpy()

# Визначення 15 оптимальних локацiй
flat_idx = np.argsort(p_final.ravel())[::-1][:15]
rows, cols = np.unravel_index(flat_idx, (GRID_H, GRID_W))
opt_lons = west + cols / GRID_W * (east - west)
opt_lats = north - rows / GRID_H * (north - south)

# Збереження растра придатностi у форматi GeoTIFF
output_tif = "kyiv_logistics_heatmap.tif"
with rasterio.open(
    output_tif, 'w',
    driver='GTiff',
    height=GRID_H, width=GRID_W,
    count=1, dtype='float32',
    crs=CRS.from_epsg(4326),
    transform=transform
) as dst:
    dst.write(p_final.astype(np.float32), 1)

# Збереження оптимальних точок у форматi shapefile
gdf_result = gpd.GeoDataFrame(
    {'score': p_final[rows, cols]},
    geometry=[Point(lon, lat)
              for lon, lat in zip(opt_lons, opt_lats)],
    crs='EPSG:4326'
)
output_shp = "kyiv_logistics_centers_multi.shp"
gdf_result.to_file(output_shp)

print(f" Збережено теплову карту: {output_tif}")
print(f" Збережено оптимальнi точки: {output_shp}")
print("\nГотово! Завантажте файли у QGIS для вiзуалiзацiї.")
