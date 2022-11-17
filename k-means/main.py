import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, \
    adjusted_mutual_info_score, silhouette_score

from kmeans import k_means
from kmeans import get_logger


def compute_metrics(x_train, y_train, y_pred_train):
    return {
        "ARI": adjusted_rand_score(y_train, y_pred_train),
        "Completeness": completeness_score(y_train, y_pred_train),
        "Homogeneity": homogeneity_score(y_train, y_pred_train),
        "V-measure": v_measure_score(y_train, y_pred_train),
        "AMI": adjusted_mutual_info_score(y_train, y_pred_train),
        "Silhouette": silhouette_score(x_train, y_pred_train)
    }


def draw_data(X_test, y_test, name_png: str, name: str):
    tmp = pd.DataFrame(data=X_test)
    tmp['y'] = np.asarray(y_test)
    g = sns.pairplot(tmp, hue='y', markers='*')
    g.savefig(f'./datasets/{name}/{name_png}')


logger = get_logger("CLUSTERINZATION")


def clusterization(df, goal_field, name, count_of_clusters, draw_plots=False):
    logger.info(f"Кластеризация датасета {name}. Кол-во кластеров {count_of_clusters}")
    observations = df.drop([goal_field], axis=1).to_numpy()
    CLASSES = np.sort(df[goal_field].unique())
    for index, c in enumerate(CLASSES):
        df.loc[df[goal_field] == c, goal_field] = index

    X_train = observations
    y_true = df[goal_field]

    logger.info("Поиск центра кластеров.")
    centroids, y_train_pred, inertia = k_means(X_train, count_of_clusters)
    logger.info("Вычисление центров кластеров завершено.")

    logger.info("Подсчет метрик...")
    metrics_on_train = compute_metrics(X_train, y_true, y_train_pred)
    metrics_on_train['sum of squared distance'] = inertia

    if draw_plots:
        draw_data(X_train, y_train_pred, f'{name}-{count_of_clusters}-classes-train-pred.png', name)

    return metrics_on_train


def compute_optim_k(df, goal_field, name, end=10, draw_plots=False):
    metrics = []
    for i in range(2, end):
        data = df.copy()
        metrics_on_train = clusterization(data, goal_field, name, count_of_clusters=i, draw_plots=draw_plots)
        metrics_on_train['count_of_clusters'] = i
        metrics.append(round(pd.DataFrame([metrics_on_train]), 4))
        print(i)

    metrics = pd.concat(metrics, ignore_index=True)
    metrics['sum of squared distance'] = round(metrics['sum of squared distance'] / metrics['sum of squared distance'].max(), 4)

    print(f'{name} metrics')
    print(metrics)
    metrics.to_csv(f'./datasets/{name}/metric.csv')

    df = metrics.melt('count_of_clusters', var_name='cols', value_name='vals')
    g = sns.catplot(x="count_of_clusters", y="vals", hue='cols', data=df, kind='point')
    g.savefig(f'{name}-metrics.png')


def main():
    print('Wine')
    df = pd.read_csv('datasets/Wine.csv')
    df_filtered = df[['Wine', 'Color.int', 'Hue', 'OD', 'Proline']]
    # data = df.copy()
    goal_field = 'Wine'
    name = 'Wine'
    X = df_filtered.drop('Wine', axis=1)
    y_true = df["Wine"].to_numpy()
    draw_data(X, y_true, 'Wine.png', name)
    compute_optim_k(df_filtered, goal_field=goal_field, name=name, end=10, draw_plots=True)
    # metrics_on_train = clusterization(data, goal_field, name, count_of_clusters=3, draw_plots=True)
    # print(pd.DataFrame([metrics_on_train]).round(2))


if __name__ == "__main__":
    main()
