import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import tushare as ts
import pandas as pd
from sklearn.covariance import GraphLassoCV
from sklearn.cluster import affinity_propagation
from sklearn import manifold
from matplotlib.collections import LineCollection


# 数据规整函数，用于对获取的df进行数据处理
def preprocess_data(stock_df,min_K_num=2246):
    '''preprocess the stock data.
    Notice: min_K_num: the minimum stock K number.
        because some stocks was halt trading in this time period,
        the some K data was missing.
        if the K data number is less than min_K_num, the stock is discarded.'''
    df=stock_df.copy()
    df['diff']=df.close-df.open  # 此处用收盘价与开盘价的差值做分析
    df.drop(['open','close','high','low'],axis=1,inplace=True)
    result_df=None
    #下面一部分是将不同的股票diff数据整合为不同的列，列名为股票代码
    for name, group in df[['date','diff']].groupby(df.code):
        if len(group.index)<min_K_num: continue
        if result_df is None:
            result_df=group.rename(columns={'diff':name})
        else:
            result_df=pd.merge(result_df,
                                group.rename(columns={'diff':name}),
                                on='date',how='inner') # 一定要inner，要不然会有很多日期由于股票停牌没数据
    result_df.drop(['date'],axis=1,inplace=True)
    result_df.stack(level=0)
    # 然后将股票数据DataFrame转变为np.ndarray
    stock_dataset=np.array(result_df)
    # 数据归一化，此处使用相关性而不是协方差的原因是在结构恢复时更高效
    stock_dataset/=np.std(stock_dataset,axis=0)
    return stock_dataset,result_df.columns.tolist()

def cluster(stock_dataset,selected_stocks,sz50_df2):
    #根据相关性学习图结构
    edge_model = GraphLassoCV()
    edge_model.fit(stock_dataset)
    #根据协方差进行AP聚类，取相似度中值，cluster_centers_indices_
    cluster_centers_indices_, labels = affinity_propagation(edge_model.covariance_)
    #print(cluster_centers_indices_)
    n_labels = max(labels)
    # 对股票进行了聚类，labels里面是每只股票对应的类别标号
    print('Stock Clusters: {}'.format(n_labels + 1))  # 10，即得到10个类别
    # 获取质心股票代码
    mass=[]
    for n in cluster_centers_indices_:
        mass.append(selected_stocks[n])
    #获取股票名称
    center_name= sz50_df2.loc[mass, :].name.tolist()
    #写入文件
    center=pd.DataFrame(np.column_stack((mass,center_name)),columns=['code','name'])
    center.to_csv(str('./Cluster/center.csv'))
    for i in range(n_labels + 1):
        # 下面打印出股票名称，便于观察
        stocks = np.array(selected_stocks)[labels == i].tolist()
        names = sz50_df2.loc[stocks, :].name.tolist()
        print('Cluster: {}----> stocks: {}'.format(str(i), ','.join(names)))
        result=pd.DataFrame(np.column_stack((stocks,names)),columns=['code','name'])
        result.to_csv(str('./Cluster/cluster '+str(i)+'.csv'))
        #visual_stock_relationship(stock_dataset, edge_model, labels, names)

#每一个节点代表一只股票，旁边有股票名称，节点的颜色表示该股票所属类别的种类，用节点颜色来区分股票所属簇群
#GraphLassoCV图结构模型中的稀疏逆协方差信息用节点之间的线条来表示，线条越粗，表示股票之间的关联性越强
#股票在图形中的位置是由2D嵌套算法来决定的，距离越远，表示其相关性越弱，簇间距离越远
def visual_stock_relationship(dataset,edge_model,labels,stock_names):
    node_position_model = manifold.LocallyLinearEmbedding( n_components=2, eigen_solver='dense', n_neighbors=6)
    embedding = node_position_model.fit_transform(dataset.T).T
    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')
    # Display a graph of the partial correlations\n",
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)
    # Plot the nodes using the coordinates of our embedding\n",
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,cmap=plt.cm.nipy_spectral)
    # Plot the edges\n",
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::\n",
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)\n",
    segments = [[embedding[:, start], embedding[:, stop]]
    for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,zorder=0, cmap=plt.cm.hot_r,norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)
    # Add a label to each node. The challenge here is that we want to\n",
    # position the labels to avoid overlap with other labels\n",
    n_labels=max(labels)
    for index, (name, label, (x, y)) in enumerate(zip(stock_names, labels, embedding.T)):
        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .001
        else:
            horizontalalignment = 'right'
            x = x - .001
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .001
        else:
            verticalalignment = 'top'
            y = y - .001
        plt.text(x, y, name, size=10,fontproperties = 'SimHei',horizontalalignment=horizontalalignment,verticalalignment=verticalalignment,bbox=dict(facecolor='w',
    edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
    alpha=.6)),
    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
                 embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
              embedding[1].max() + .03 * embedding[1].ptp())
    plt.show()


if __name__ == '__main__':
    #获取沪深300代码表
    sz50_df = ts.get_hs300s()
    sz50_df2 = sz50_df.set_index('code')
    #读取已筛选的数据
    filtered = pd.read_csv("./filtered_data.csv", converters={'code': str})
    filtered[u'code'] = filtered[u'code'].apply(lambda x: x[3:])
    #处理数据，使用变化幅度聚类
    stock_dataset, selected_stocks = preprocess_data(filtered, min_K_num=2246)
    #聚类并画图
    cluster(stock_dataset, selected_stocks,sz50_df2)