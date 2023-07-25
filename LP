import pandas as pd
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Функция для чтения данных из JSON-файла с визуализацией прогресса
def read_json_file_with_progress(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Путь к папке с архивами JSON-файлов
archive_folder = 'Archive_json'

# Список для хранения данных из всех файлов
all_data = []

# Чтение данных из CSV файла
csv_file_path = 'UsersWithoutServers1.csv'
csv_data = pd.read_csv(csv_file_path)

# Удаление строк с отсутствующими значениями Latency
csv_data.dropna(subset=['Latency'], inplace=True)

# Обход папок с датами и чтение данных из JSON-файлов с визуализацией прогресса
for date_folder in tqdm(os.listdir(archive_folder), desc="Чтение файлов", unit="файл"):
    date_folder_path = os.path.join(archive_folder, date_folder)
    if os.path.isdir(date_folder_path):
        for file_name in tqdm(os.listdir(date_folder_path), desc=date_folder, unit="файл"):
            file_path = os.path.join(date_folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.json'):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # Извлекаем значение метрики из данных JSON
                    metric_value = data.get('ReleemMetrics', {}).get('Questions', None)
                    if metric_value is not None:
                        # Добавляем новую метрику в DataFrame csv_data
                        csv_data.loc[csv_data['timestamp'] == int(data['Timestamp']), 'Questions'] = metric_value

# Выполнение кластеризации на основе метрики Latency
X = csv_data[['Latency']]
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Добавим метку кластера в DataFrame
csv_data['Cluster'] = kmeans.labels_

# Визуализация результатов кластеризации
for cluster_num in range(3):
    cluster_data = csv_data[csv_data['Cluster'] == cluster_num]
    plt.scatter(cluster_data.index, cluster_data['Latency'], c=cluster_data['Latency'], cmap='viridis', label=f'Cluster {cluster_num}', alpha=0.7)

plt.xlabel('Data Point Index')
plt.ylabel('Latency')
plt.title('Clustered Load on Database Servers')
plt.legend(title='Cluster', loc='upper right')

# Вывод описания каждого кластера
for cluster_num in range(3):
    cluster_data = csv_data[csv_data['Cluster'] == cluster_num]
    cluster_mean_latency = cluster_data['Latency'].mean()
    print(f'Cluster {cluster_num} - Mean Latency: {cluster_mean_latency:.2f} ms')

plt.colorbar(label='Latency (ms)')
plt.show()
